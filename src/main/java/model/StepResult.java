package model;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@AllArgsConstructor
@Getter
@Setter
public class StepResult {
    public final State state;
    public final double reward;
    public final boolean done;

}
