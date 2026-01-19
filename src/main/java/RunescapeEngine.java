
import model.EpisodeStats;
import model.State;
import model.StepResult;
import model.Transition;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RunescapeEngine {

    //ACTIONS
    int STAND = 0;
    int ATTACK = 1;

    // EPISODE RELATED
    private int EPISODES = 1000_000;
    private int LOG_EVERY = 10_000;
    private long SEED = 1234L;

    //HYPER PARAMS
    private double CRITIC_ALPHA = 0.06;
    private double ACTOR_ALPHA = 0.04;
    private double GAMMA = 0.99;

    //PPO
    private int EPOCHS = 4;

    double alphaTrunk = 0.5 * ACTOR_ALPHA;

    //ANN
    private int ANN_INPUT_COUNT = 17;
    private int ANN_ACTION_COUNT = 2;
    private int ANN_NEURONS = 32;
    TinyActorCriticANN tinyActorCriticANN;
    StateVectorizer stateVectorizer;

    //GAME ENGINE RELATED
    private int failedStands = 0;
    private int hpRegeneratedFromStand = 0;
    private int standOutOfCombatWhileInjured = 0;
    private int standOutOfCombatAtFullHp = 0;

    @Test
    public void execute(){
        tinyActorCriticANN = new TinyActorCriticANN(ANN_INPUT_COUNT, ANN_NEURONS, ANN_ACTION_COUNT);
        stateVectorizer = new StateVectorizer(new StateVectorizer.Bounds());
        runSimulation(EPISODES, SEED);
    }

    private void runSimulation(int episodes, long seed) {
        Random rng = new Random(seed);

        double totalAvgR = 0.0;

        long stepsWin = 0;
        double rewardWin = 0.0;

        long standWin = 0;
        long attackWin = 0;

        long failedStandsWin = 0;
        long regenHpWin = 0;

        long xpLeftWin = 0;
        long levelsWin = 0;

        long killsWin = 0;
        long deathsWin = 0;

        long standWhileInjured = 0;
        long standWhileFullHp = 0;

            for (int ep = 1; ep <= episodes; ep++) {
                hpRegeneratedFromStand = 0;
                failedStands = 0;
                standOutOfCombatWhileInjured = 0;
                standOutOfCombatAtFullHp = 0;

                EpisodeStats es = executeEpisode(rng);

                stepsWin += es.steps;
                rewardWin += es.totalReward;

                standWin += es.standCount;
                attackWin += es.attackCount;

                failedStandsWin += es.failedStands;
                regenHpWin += es.hpRegeneratedFromStand;

                xpLeftWin += es.xpGained;
                levelsWin += es.levelsGained;

                killsWin += es.endKills;
                deathsWin += es.endDeaths;

                standWhileInjured += es.standOutOfCombatWhileInjured;
                standWhileFullHp += es.standOutOfCombatAtFullHp;

                if (ep % LOG_EVERY == 0) {
                    double n = (double) LOG_EVERY;

                    double avgR = rewardWin / n;
                    double avgSteps = stepsWin / n;

                    double avgAttack = attackWin / n;
                    double avgStand = standWin / n;

                    double avgFailedStands = failedStandsWin / n;
                    double avgRegenHp = regenHpWin / n;

                    double avgXpLeft = xpLeftWin / n;
                    double avgLevels = levelsWin / n;

                    double avgKills = killsWin / n;
                    double avgDeaths = deathsWin / n;
                    double kd = killsWin / (double) Math.max(1L, deathsWin); // global window K/D

                    double avgStandWhileInjured = standWhileInjured / n;
                    double avgStandWhileFullHp = standWhileFullHp / n;

                    totalAvgR += avgR;

                    System.out.printf(
                            "[Ep %,d] avgR=%.3f avgSteps=%.1f | K/D=%.2f (K=%.2f D=%.2f) | act: atk=%.1f stand=%.1f | standFail=%.2f regenHp=%.2f | standInjured=%.2f standFull=%.2f | lvl+=%.2f xpLeft=%.1f%n",
                            ep, avgR, avgSteps,
                            kd, avgKills, avgDeaths,
                            avgAttack, avgStand,
                            avgFailedStands, avgRegenHp,
                            avgStandWhileInjured, avgStandWhileFullHp,
                            avgLevels, avgXpLeft
                    );

                    // reset window
                    stepsWin = 0;
                    rewardWin = 0.0;
                    standWin = 0;
                    attackWin = 0;
                    failedStandsWin = 0;
                    regenHpWin = 0;
                    xpLeftWin = 0;
                    levelsWin = 0;
                    killsWin = 0;
                    deathsWin = 0;
                    standWhileInjured = 0;
                    standWhileFullHp = 0;
                }
        }
    }

    private EpisodeStats executeEpisode(Random rng) {
        // Trajectory buffer. PPO collects an entire episode of experiences
        // before any learning happens. Each transition stores the state,
        // action, probability at selection time, reward, and next state.
        List<Transition> trajectory = new ArrayList<>();

        EpisodeStats es = new EpisodeStats();
        State currentState = initState();
        
        // Sample action and store its probability. This oldProb will be
        // stored in the trajectory and later used to compute the importance
        // sampling ratio during PPO updates.
        ActionSample sample = chooseAction(currentState, rng);
        int currentAction = sample.action;
        double oldProb = sample.oldProb;

        // initiate game
        while(currentState.getKills() < 30 && currentState.getDeaths() < 30 && es.steps < 999){
            //execute step
            StepResult stepResult = step(currentState, currentAction, rng);

            // Store transition for later batch update. The oldProb is crucial
            // for PPO as it will be compared against the current probability
            // during the epoch loop to compute the importance sampling ratio.
            trajectory.add(new Transition(
                    stateVectorizer.encode(currentState),
                    currentAction,
                    oldProb,
                    stepResult.reward,
                    stateVectorizer.encode(stepResult.state),
                    stepResult.done
            ));

            // Stats
            es.steps++;
            es.totalReward += stepResult.getReward();
            if (currentAction == ATTACK) es.attackCount++;
            else if (currentAction == STAND) es.standCount++;

            // update prev/current state
            currentState = stepResult.state; //update our state
            if (currentState.getKills() >= 30 || currentState.getDeaths() >= 30 || es.steps >= 999) break;
            ActionSample sampleNewAction = chooseAction(currentState, rng);
            currentAction = sampleNewAction.action; //chose next action
            oldProb = sampleNewAction.oldProb; // update old prob?
        }

        // Compute advantages using frozen critic snapshot. All transitions
        // get their advantage calculated before any weight updates occur.
        // This provides consistent training signal across the batch.
        computeAdvantages(trajectory);
        
        // Multiple epochs over the same trajectory. This improves sample
        // efficiency by reusing collected experiences. The PPO clipping
        // mechanism prevents instability as the policy drifts across epochs.
        // By epoch 3 or 4, many experiences will be clipped because the
        // importance ratio has moved outside the allowed range.
        for(int i = 0; i < EPOCHS; i++) {
            for (Transition t : trajectory) {
                tinyActorCriticANN.update(
                        t.state,
                        t.action,
                        t.oldProb,
                        t.advantage,
                        CRITIC_ALPHA,
                        ACTOR_ALPHA,
                        alphaTrunk
                );
            }
        }

        es.endKills = currentState.getKills();
        es.endDeaths = currentState.getDeaths();
        es.xpGained = currentState.getXpCollected();
        es.levelsGained = currentState.getLevelsIncreased();
        es.failedStands += failedStands;
        es.hpRegeneratedFromStand += hpRegeneratedFromStand;
        es.standOutOfCombatAtFullHp += standOutOfCombatAtFullHp;
        es.standOutOfCombatWhileInjured += standOutOfCombatWhileInjured;
        return es;
    }

    /**
     * Computes TD(0) advantage for each transition in the trajectory.
     * 
     * Advantage measures how much better the actual outcome was compared
     * to what the critic predicted. Positive advantage means the action
     * led to better results than expected. Negative means worse.
     * 
     * All advantages are computed using the same critic snapshot before
     * any updates. This frozen baseline reduces variance in training.
     */
    void computeAdvantages(List<Transition> trajectory) {
        for (Transition t : trajectory) {
            double vS = tinyActorCriticANN.value(t.state);
            double vNext = t.done ? 0.0 : tinyActorCriticANN.value(t.nextState);

            t.advantage = t.reward + GAMMA * vNext - vS;
            t.returnTarget = t.reward + GAMMA * vNext;
        }
    }

    public ActionSample chooseAction(State state, Random rng) {
        double[] p = tinyActorCriticANN.policyProbs(stateVectorizer.encode(state));
        double r = rng.nextDouble();
        double c = 0.0;
        for (int a = 0; a < p.length; a++) {
            c += p[a];
            if (r <= c) {
                return new ActionSample(a, p[a]);
            }
        }

        int last = p.length - 1;
        return new ActionSample(last, p[last]);
    }

    public StepResult step(State state, int action, Random rng) {
        State ns = copyOf(state);
        double r = 0.0;

        if (action == ATTACK) {
            ns.setInCombat(1);

            int playerDmg = rollPlayerHit(ns, rng);

            // HIT NPC
            // NPC kill
            if(playerDmg >= ns.getNpcCurrentHp()){
                r += 1;
                ns.setKills(ns.getKills() + 1);
                ns.setInCombat(0);
                applyLevelsAndExperience(ns, ns.getNpcCurrentHp());
                // NPS respawns:
                ns.setNpcCurrentHp(ns.getNpcMaxHp());
                ns.setStands(0);
            }else{
                // Apply dmg to npc
                ns.setNpcCurrentHp(ns.getNpcCurrentHp() - playerDmg);
                // Apply experience
                applyLevelsAndExperience(ns, playerDmg);
            }

            //NPC HITS PLAYER
            // if NPC is dead let's skip NPC turn
            if (ns.getInCombat() == 1) {
                int npcDmg = rollNpcHit(ns, rng);
                // Player was killed
                if (npcDmg >= ns.getCurrentHp()) {
                    r -= 1;
                    ns.setInCombat(0);
                    // Player respawns
                    ns.setCurrentHp(ns.getMaxHp());
                    ns.setDeaths(ns.getDeaths() + 1);
                    ns.setStands(0);
                    //NPC resets to full hp after death (not exactly how it works in rs)
                    ns.setNpcCurrentHp(ns.getNpcMaxHp());
                } else {
                    ns.setCurrentHp(ns.getCurrentHp() - npcDmg);
                }
            }
        } else if (action == STAND) {
            // if player is in combat generate npc hit
            if(ns.getInCombat() == 1){
                int npcDmg = rollNpcHit(ns, rng);
                r -= 0.01;
                if (npcDmg >= ns.getCurrentHp()) {
                    r -= 1;
                    ns.setInCombat(0);
                    ns.setDeaths(ns.getDeaths() + 1);
                    // Player respawns
                    ns.setCurrentHp(ns.getMaxHp());
                    ns.setStands(0);
                    this.failedStands++;
                    //NPC resets to full hp after death (not exactly how it works in rs)
                    ns.setNpcCurrentHp(ns.getNpcMaxHp());
                } else {
                    ns.setCurrentHp(ns.getCurrentHp() - npcDmg);
                }
            }else{
                ns.setStands(ns.getStands() + 1);
                if(ns.getCurrentHp() < ns.getMaxHp()) {
                    standOutOfCombatWhileInjured += 1;
                    ns.setCurrentHp(ns.getCurrentHp() + 1);
                    hpRegeneratedFromStand += 1;
                }else{
                    standOutOfCombatAtFullHp += 1;
                }
            }
        }
        if(ns.getKills() >= 30 || ns.getDeaths() >= 30){
            return new StepResult(ns, r, true);
        }else{
            return new StepResult(ns, r, false);
        }
    }


    private State applyLevelsAndExperience(State s, int hit) {
        if (hit <= 0) return s;

        // 1) Gain XP from damage
        int gainedXp = getXpCountFromHit(hit); // e.g. hit * 4
        s.setXpCollected(s.getXpCollected() + gainedXp);

        // 2) Level up loop (can level multiple times from one big hit)
        while (s.getXpCollected() >= xpToNext(s)) {
            int need = xpToNext(s);
            s.setXpCollected(s.getXpCollected() - need);
            s.setLevelsIncreased(s.getLevelsIncreased() + 1);

            // 3) Apply the level-up to the skill based on fight style
            int style = s.getFightStyle();
            if (style == 0) { // STAB -> Attack
                s.setAttackLvl(s.getAttackLvl() + 1);
            } else if (style == 1) { // SLASH -> Strength
                s.setStrLvl(s.getStrLvl() + 1);
            } else if (style == 2) { // DEF HIT -> Defence
                s.setDefenceLvl(s.getDefenceLvl() + 1);
            } else {
                // Unknown style: default to Attack (or throw)
                s.setAttackLvl(s.getAttackLvl() + 1);
            }
        }

        return s;
    }

    private int rollNpcHit(State s, Random rng) {
        double hitChance =
                s.getNpcAttack() / (double) (s.getNpcAttack() + s.getDefenceLvl() + 1);

        if (rng.nextDouble() < hitChance) {
            int maxHit = Math.max(1, s.getNpcStr() / 2);
            return 1 + rng.nextInt(maxHit);
        }
        return 0;
    }

    private int getXpCountFromHit(int hit){
        return hit * 4;
    }
    private int getHitpointsXpFromHit(int hit) {
        // ~1.33 per damage ≈ 4/3
        return (hit * 4) / 3; // integer division
    }

    private int xpToNext(State s) {
        return 50 * (s.getLevelsIncreased() + 1);
    }

    private int rollPlayerHit(State s, Random rng) {
        double hitChance =
                s.getAttackLvl() / (double) (s.getAttackLvl() + s.getNpcDef() + 1);

        if (rng.nextDouble() < hitChance) {
            int maxHit = Math.max(1, s.getStrLvl() / 2);
            return 1 + rng.nextInt(maxHit);
        }
        return 0; // miss
    }


    private State initState() {
        State initialState = new State(
                10,
                10,
                10,
                10,
                10,
                30,
                30,
                5,
                4,
                5,
                0,
                0,
                0,
                0,
                0,
                0,
                0);
        return initialState;
    }

    private State copyOf(State s) {
        return new State(
                s.getMaxHp(),
                s.getCurrentHp(),
                s.getStrLvl(),
                s.getAttackLvl(),
                s.getDefenceLvl(),
                s.getNpcMaxHp(),
                s.getNpcCurrentHp(),
                s.getNpcAttack(),
                s.getNpcDef(),
                s.getNpcStr(),
                s.getInCombat(),
                s.getXpCollected(),
                s.getLevelsIncreased(),
                s.getFightStyle(),
                s.getKills(),
                s.getDeaths(),
                s.getStands()
        );
    }

    public record ActionSample(int action, double oldProb) {}
}
