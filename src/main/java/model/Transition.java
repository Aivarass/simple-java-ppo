package model;

public class Transition {
    public final double[] state;
    public final int action;
    public final double oldProb;
    public final double reward;
    public final double[] nextState;
    public final boolean done;
    public double advantage;
    public double returnTarget;

    public Transition(double[] state, int action, double oldProb,
                      double reward, double[] nextState, boolean done) {
        this.state = state;
        this.action = action;
        this.oldProb = oldProb;
        this.reward = reward;
        this.nextState = nextState;
        this.done = done;
    }
}