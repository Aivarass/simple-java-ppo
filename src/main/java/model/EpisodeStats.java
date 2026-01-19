package model;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Setter
@Getter
@NoArgsConstructor
public class EpisodeStats {
    public int steps;
    public double totalReward;
    public int endKills;
    public int endDeaths;
    public int standCount;
    public int attackCount;
    public int failedStands;
    public int hpRegeneratedFromStand;
    public int xpGained;
    public int levelsGained;
    public int standOutOfCombatWhileInjured;
    public int standOutOfCombatAtFullHp;
}
