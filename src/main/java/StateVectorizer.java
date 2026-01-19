import model.State;

public final class StateVectorizer {

    public static final class Bounds {
        public int hpMaxMax = 99;       // max possible maxHp in your sim (player)
        public int npcHpMaxMax = 999;   // max possible npcMaxHp in your sim
        public int lvlMax = 99;         // player skill level cap
        public int npcStatMax = 999;    // npc stats cap
        public int xpPivot = 10_000;    // soft-saturation pivot
        public int lvlIncPivot = 10;    // soft-saturation pivot
        public int fightStyleMax = 2;   // 0..2
        public int killsPivot = 30;
        public int deathsPivot = 30;
        public int standsMax = 3;
    }

    private final Bounds b;

    public StateVectorizer(Bounds bounds) {
        this.b = bounds;
    }

    public int featureCount() {
        return 17;
    }

    public double[] encode(State s) {
        double[] x = new double[featureCount()];
        int i = 0;

        // ---- Player ----
        x[i++] = mm11(s.getMaxHp(),      1, b.hpMaxMax);
        x[i++] = mm11(s.getCurrentHp(),  0, Math.max(1, s.getMaxHp()));

        x[i++] = mm11(s.getStrLvl(),     1, b.lvlMax);
        x[i++] = mm11(s.getAttackLvl(),  1, b.lvlMax);
        x[i++] = mm11(s.getDefenceLvl(), 1, b.lvlMax);

        // ---- NPC ----
        x[i++] = mm11(s.getNpcMaxHp(),     1, b.npcHpMaxMax);
        x[i++] = mm11(s.getNpcCurrentHp(), 0, Math.max(1, s.getNpcMaxHp()));

        x[i++] = mm11(s.getNpcAttack(), 1, b.npcStatMax);
        x[i++] = mm11(s.getNpcDef(),    1, b.npcStatMax);
        x[i++] = mm11(s.getNpcStr(),    1, b.npcStatMax);

        // ---- Context ----
        x[i++] = (s.getInCombat() == 1) ? 1.0 : -1.0;

        // ---- Progression (soft-saturated then mapped to [-1,1]) ----
        x[i++] = soft11(s.getXpCollected(), b.xpPivot);
        x[i++] = soft11(s.getLevelsIncreased(), b.lvlIncPivot);

        // fightStyle: 0..2 -> [-1,1]
        x[i++] = mm11(s.getFightStyle(), 0, b.fightStyleMax);

        x[i++] = soft11(s.getKills(),  b.killsPivot);
        x[i++] = soft11(s.getDeaths(), b.deathsPivot);

        // stands counter is small, scale to [-1,1]
        x[i++] = mm11(s.getStands(), 0, b.standsMax);

        // Safety check for dev/debug
        if (i != x.length) {
            throw new IllegalStateException("FeatureCount mismatch: filled=" + i + " len=" + x.length);
        }

        return x;
    }

    private static double mm11(double v, double lo, double hi) {
        if (hi <= lo) return 0.0;
        double vc = clamp(v, lo, hi);
        double r01 = (vc - lo) / (hi - lo);
        return 2.0 * r01 - 1.0;
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static double soft11(double x, double pivot) {
        double p = Math.max(1e-9, pivot);
        double z = Math.max(0.0, x) / p;
        double y01 = z / Math.sqrt(1.0 + z * z);
        return 2.0 * y01 - 1.0;
    }
}