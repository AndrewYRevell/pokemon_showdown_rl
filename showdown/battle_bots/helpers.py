import constants


def format_decision(battle, decision):
    # Formats a decision for communication with Pokemon-Showdown
    # If the pokemon can mega-evolve, it will
    # If the move can be used as a Z-Move, it will be

    if decision.startswith(constants.SWITCH_STRING + " "):
        switch_pokemon = decision.split("switch ")[-1]
        for i in range(len(battle.user.reserve)):
            pkmn = battle.user.reserve[i]
            if pkmn.name == switch_pokemon:
                try:
                    message = "/switch {}".format(pkmn.index)
                    break
                except:
                    #find indexes of all other pokemon and find the missing
                    ind = []
                    for n in range(len(battle.user.reserve)):
                        p = battle.user.reserve[n]
                        if n == i: continue
                        ind.append(p.index)
                    ind = sorted(set(range(2, 6 + 1)).difference(ind))[0] #find missing index
                    message = "/switch {}".format(ind)
                    break
        else:
            raise ValueError("Tried to switch to: {}".format(switch_pokemon))
    else:
        message = "/choose move {}".format(decision)
        if battle.user.active.can_mega_evo:
            message = "{} {}".format(message, constants.MEGA)
        elif battle.user.active.can_ultra_burst:
            message = "{} {}".format(message, constants.ULTRA_BURST)

        # only dynamax on last pokemon
        #if battle.user.active.can_dynamax and all(p.hp == 0 for p in battle.user.reserve):
        #    message = "{} {}".format(message, constants.DYNAMAX)

        #if battle.user.active.get_move(decision).can_z:
        #    message = "{} {}".format(message, constants.ZMOVE)

    return [message, str(battle.rqid)]
