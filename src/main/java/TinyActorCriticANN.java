import java.util.Arrays;
import java.util.Random;

/**
 * A shallow Actor Critic neural network implementing PPO from scratch.
 * 
 * Architecture: shared trunk with tanh hidden layer, branching into
 * actor head (softmax policy) and critic head (value estimate).
 * 
 * The update method implements the core PPO algorithm including
 * importance sampling ratio, clipped surrogate objective, and
 * entropy regularization.
 */
public class TinyActorCriticANN {

    // Entropy bonus coefficient. Encourages exploration by rewarding
    // uncertainty in action selection. This gradient always flows,
    // even when the policy gradient is clipped.
    private final double ENTROPY_BETA = 0.005;

    private final double ENTROPY_EPS = 1e-12;
    
    // PPO clipping range. When the importance ratio goes outside
    // [1 - epsilon, 1 + epsilon], the policy gradient is zeroed out.
    // This prevents destructive updates when the policy has drifted
    // too far from its state during trajectory collection.
    private final double PPO_EPSILON = 0.2;

    private final int inputDim;
    private final int hiddenUnits;
    private final int actionCount;

    // Trunk: hidden = tanh( Wih * x + bh )
    private final double[][] wInputHidden; // [H][D]
    private final double[] bHidden;        // [H]

    // Actor head: logits = Wa * hidden + ba
    private final double[][] wHiddenActor; // [A][H]
    private final double[] bActor;         // [A]

    // Critic head: value = Wv * hidden + bv
    private final double[] wHiddenValue;   // [H]

    private double bValueMutable;

    private final Random rng;

    private final double[] wEnt;

    // ---- Scratch buffers (avoid allocations each step) ----
    private final double[] hidden;     // [H]
    private final double[] logits;     // [A]
    private final double[] probs;      // [A]

    public TinyActorCriticANN(int inputDim, int hiddenUnits, int actionCount, long seed) {
        if (inputDim <= 0 || hiddenUnits <= 0 || actionCount <= 1) {
            throw new IllegalArgumentException("inputDim>0, hiddenUnits>0, actionCount>1 required");
        }
        this.inputDim = inputDim;
        this.hiddenUnits = hiddenUnits;
        this.actionCount = actionCount;
        this.rng = new Random(seed);

        this.wInputHidden = new double[hiddenUnits][inputDim];
        this.bHidden = new double[hiddenUnits];

        this.wHiddenActor = new double[actionCount][hiddenUnits];
        this.bActor = new double[actionCount];

        this.wHiddenValue = new double[hiddenUnits];
        this.bValueMutable = 0.0;

        this.hidden = new double[hiddenUnits];
        this.logits = new double[actionCount];
        this.probs = new double[actionCount];

        this.wEnt = new double[actionCount];

        initWeightsXavier();
    }

    public TinyActorCriticANN(int inputDim, int hiddenUnits, int actionCount) {
        this(inputDim, hiddenUnits, actionCount, System.nanoTime());
    }

    private void initWeightsXavier() {
        // Xavier-ish init for tanh trunk
        double limitIH = Math.sqrt(6.0 / (inputDim + hiddenUnits));
        for (int h = 0; h < hiddenUnits; h++) {
            for (int d = 0; d < inputDim; d++) {
                wInputHidden[h][d] = uniform(-limitIH, limitIH);
            }
            bHidden[h] = 0.0;
        }

        // Actor head
        double limitHA = Math.sqrt(6.0 / (hiddenUnits + actionCount));
        for (int a = 0; a < actionCount; a++) {
            for (int h = 0; h < hiddenUnits; h++) {
                wHiddenActor[a][h] = uniform(-limitHA, limitHA);
            }
            bActor[a] = 0.0;
        }

        // Critic head
        double limitHV = Math.sqrt(6.0 / (hiddenUnits + 1.0));
        for (int h = 0; h < hiddenUnits; h++) {
            wHiddenValue[h] = uniform(-limitHV, limitHV);
        }
        bValueMutable = 0.0;
    }

    private double uniform(double min, double max) {
        return min + (max - min) * rng.nextDouble();
    }

    public double[] policyProbs(double[] x) {
        forward(x);
        return Arrays.copyOf(probs, probs.length);
    }

    public double policyProb(double[] x, int action) {
        forward(x);
        return probs[action];
    }

    public int sampleAction(double[] x) {
        forward(x);
        double r = rng.nextDouble();
        double cdf = 0.0;
        for (int a = 0; a < actionCount; a++) {
            cdf += probs[a];
            if (r <= cdf) return a;
        }
        return actionCount - 1; // numerical safety
    }

    public double value(double[] x) {
        forwardHiddenOnly(x);
        return computeValueFromHidden();
    }


    /**
     * PPO update step. Takes a pre computed advantage from the trajectory
     * and updates critic, actor, and trunk weights.
     * 
     * The key PPO mechanism is the importance sampling ratio and clipping.
     * When the policy has changed too much since the action was taken,
     * the policy gradient is suppressed to prevent instability.
     */
    public double update(double[] s,
                         int action,
                         double oldProb,
                         double advantage,
                         double alphaCritic,
                         double alphaActor,
                         double alphaTrunk) {

        forward(s);

        // Importance sampling ratio. Compares current policy probability
        // to the probability when the action was originally taken.
        // If ratio > 1, policy now favors this action more than before.
        // If ratio < 1, policy now favors this action less than before.
        double piNew = probs[action];
        double ratio = piNew / (oldProb + ENTROPY_EPS);

        // Snapshot weights before update
        double[] wVsnap = Arrays.copyOf(wHiddenValue, hiddenUnits);
        double[][] wAsnap = new double[actionCount][hiddenUnits];
        for (int k = 0; k < actionCount; k++) {
            System.arraycopy(wHiddenActor[k], 0, wAsnap[k], 0, hiddenUnits);
        }

        // Clip advantage (not delta)
        double advClipped = clip(advantage, -5.0, 5.0);

        // ---- Critic update ----
        for (int h = 0; h < hiddenUnits; h++) {
            wHiddenValue[h] += alphaCritic * advClipped * hidden[h];
        }
        bValueMutable += alphaCritic * advClipped;

        // PPO clipping check. The clipped surrogate objective prevents
        // the policy from changing too much in a single update.
        // 
        // When advantage is positive (good action), we want to increase
        // its probability, but we cap how much by clipping ratio at 1 + epsilon.
        // 
        // When advantage is negative (bad action), we want to decrease
        // its probability, but we cap how much by clipping ratio at 1 - epsilon.
        // 
        // If clipped, the policy gradient becomes zero for this experience.
        boolean isClipped = (advClipped >= 0)
                ? (ratio > 1.0 + PPO_EPSILON)
                : (ratio < 1.0 - PPO_EPSILON);



        double sumW = 0.0;
        for (int j = 0; j < actionCount; j++) {
            wEnt[j] = probs[j] * (Math.log(probs[j] + ENTROPY_EPS) + 1.0);
            sumW += wEnt[j];
        }

        for (int k = 0; k < actionCount; k++) {
            // Softmax policy gradient with respect to logit k
            double pg = ((k == action) ? 1.0 : 0.0) - probs[k];
            double dH_dlogit = (-wEnt[k]) + (probs[k] * sumW);

            // Core PPO gradient. When clipped, policy learning stops for this
            // experience but entropy gradient still flows to maintain exploration.
            double policyGrad = isClipped ? 0.0 : (ratio * advClipped * pg);

            double g = policyGrad + (ENTROPY_BETA * dH_dlogit);

            double step = alphaActor * g;


            for (int h = 0; h < hiddenUnits; h++) {
                wHiddenActor[k][h] += step * hidden[h];
            }
            bActor[k] += step;
        }

        for (int h = 0; h < hiddenUnits; h++) {
            double dh_dz = 1.0 - hidden[h] * hidden[h];

            double criticPart = advClipped * wVsnap[h];

            double actorSum = 0.0;
            for (int k = 0; k < actionCount; k++) {
                double g = ((k == action) ? 1.0 : 0.0) - probs[k];
                actorSum += g * wAsnap[k][h];
            }
            double actorPart = advClipped * actorSum;

            double chain = alphaTrunk * (criticPart + actorPart) * dh_dz;

            for (int d = 0; d < inputDim; d++) {
                wInputHidden[h][d] += chain * s[d];
            }
            bHidden[h] += chain;
        }

        return advantage;
    }

    /** Full forward pass: hidden -> logits -> probs. */
    private void forward(double[] x) {
        forwardHiddenOnly(x);

        // logits
        for (int a = 0; a < actionCount; a++) {
            double z = bActor[a];
            for (int h = 0; h < hiddenUnits; h++) {
                z += wHiddenActor[a][h] * hidden[h];
            }
            logits[a] = z;
        }

        softmaxInPlace(logits, probs);
    }

    /** Only compute hidden activations (used for V(s) bootstrap). */
    private void forwardHiddenOnly(double[] x) {
        for (int h = 0; h < hiddenUnits; h++) {
            double z = bHidden[h];
            for (int d = 0; d < inputDim; d++) {
                z += wInputHidden[h][d] * x[d];
            }
            hidden[h] = Math.tanh(z);
        }
    }

    private double computeValueFromHidden() {
        double v = bValueMutable;
        for (int h = 0; h < hiddenUnits; h++) {
            v += wHiddenValue[h] * hidden[h];
        }
        return v;
    }

    private static void softmaxInPlace(double[] inLogits, double[] outProbs) {
        double max = inLogits[0];
        for (int i = 1; i < inLogits.length; i++) max = Math.max(max, inLogits[i]);

        double sum = 0.0;
        for (int i = 0; i < inLogits.length; i++) {
            double e = Math.exp(inLogits[i] - max);
            outProbs[i] = e;
            sum += e;
        }
        double inv = 1.0 / sum;
        for (int i = 0; i < outProbs.length; i++) outProbs[i] *= inv;
    }

    private static double clip(double x, double lo, double hi) {
        return Math.max(lo, Math.min(hi, x));
    }
}