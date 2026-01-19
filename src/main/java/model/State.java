package model;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class State {
    // ---- Player ----
    private int maxHp;
    private int currentHp;
    private int    strLvl;
    private int    attackLvl;
    private int    defenceLvl;

    // ---- NPC ----
    private int npcMaxHp; // npcCurrentHp / npcTotalHp
    private int npcCurrentHp;
    private int    npcAttack;
    private int    npcDef;
    private int    npcStr;

    // ---- Context ----
    private int    inCombat;             // 0 or 1

    private int    xpCollected;           // soft-saturated
    private int    levelsIncreased;       // soft-saturated
    private int    fightStyle;            // 0 STAB, 1 SLASH, 2 DEF HIT

    // -----
    private int kills;
    private int deaths;
    private int stands;


    // ---- INVENTORY ----
    /**
     * [ isEmpty,
     *   isFood, isWeapon, isArmour,
     *   heal,
     *   atkBonus, strBonus, defBonus,
     *   qty,
     *   meta ]
     */

}
