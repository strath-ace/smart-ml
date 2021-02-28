import random
from deap import gp

########################### GENETIC OPERATORS FOR MULTIPLE TREE OUTPUT   #####################################
def xmateSimple(ind1, ind2):
    """From [2] """
    i1 = random.randrange(len(ind1))
    i2 = random.randrange(len(ind2))
    ind1[i1], ind2[i2] = gp.cxOnePoint(ind1[i1], ind2[i2])
    return ind1, ind2


def xmutSimple(ind, expr, pset):
    """From [2] """
    i1 = random.randrange(len(ind))
    indx = gp.mutUniform(ind[i1], expr, pset)
    ind[i1] = indx[0]
    return ind,


def xmate(ind1, ind2):
    """From [2] and modified"""
    ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
    return ind1, ind2


def xmut(ind, expr, unipb, shrpb, inspb, pset, creator):
    """From [2] and modified. Added several mutations possibilities."""
    choice = random.random()
    try:
        if type(ind[0]) == creator.SubIndividual:
            if hasattr(pset, '__len__'):
                if choice < unipb:
                    indx1 = gp.mutUniform(ind[0], expr, pset=pset[0])
                    ind[0] = indx1[0]
                    indx2 = gp.mutUniform(ind[1], expr, pset=pset[1])
                    ind[1] = indx2[0]
                elif unipb <= choice < unipb + shrpb:
                    indx1 = gp.mutShrink(ind[0])
                    ind[0] = indx1[0]
                    indx2 = gp.mutShrink(ind[1])
                    ind[1] = indx2[0]
                elif unipb + shrpb <= choice < unipb + shrpb + inspb:
                    indx1 = gp.mutInsert(ind[0], pset=pset[0])
                    ind[0] = indx1[0]
                    indx2 = gp.mutInsert(ind[1], pset=pset[1])
                    ind[1] = indx2[0]
                else:
                    choice2 = random.random()
                    if choice2 < 0.5:
                        indx1 = gp.mutEphemeral(ind[0], "all")
                        ind[0] = indx1[0]
                        indx2 = gp.mutEphemeral(ind[1], "all")
                        ind[1] = indx2[0]
                    else:
                        indx1 = gp.mutEphemeral(ind[0], "one")
                        ind[0] = indx1[0]
                        indx2 = gp.mutEphemeral(ind[1], "one")
                        ind[1] = indx2[0]
            else:
                if choice < unipb:
                    indx1 = gp.mutUniform(ind[0], expr, pset=pset)
                    ind[0] = indx1[0]
                    indx2 = gp.mutUniform(ind[1], expr, pset=pset)
                    ind[1] = indx2[0]
                elif unipb <= choice < unipb + shrpb:
                    indx1 = gp.mutShrink(ind[0])
                    ind[0] = indx1[0]
                    indx2 = gp.mutShrink(ind[1])
                    ind[1] = indx2[0]
                elif unipb + shrpb <= choice < unipb + shrpb + inspb:
                    indx1 = gp.mutInsert(ind[0], pset=pset)
                    ind[0] = indx1[0]
                    indx2 = gp.mutInsert(ind[1], pset=pset)
                    ind[1] = indx2[0]
                else:
                    choice2 = random.random()
                    if choice2 < 0.5:
                        indx1 = gp.mutEphemeral(ind[0], "all")
                        ind[0] = indx1[0]
                        indx2 = gp.mutEphemeral(ind[1], "all")
                        ind[1] = indx2[0]
                    else:
                        indx1 = gp.mutEphemeral(ind[0], "one")
                        ind[0] = indx1[0]
                        indx2 = gp.mutEphemeral(ind[1], "one")
                        ind[1] = indx2[0]
    except AttributeError:
        if choice < unipb:
            indx1 = gp.mutUniform(ind, expr, pset=pset)
            ind = indx1[0]
        elif unipb <= choice < unipb + shrpb:
            indx1 = gp.mutShrink(ind)
            ind = indx1[0]
        elif unipb + shrpb <= choice < unipb + shrpb + inspb:
            indx1 = gp.mutInsert(ind, pset=pset)
            ind = indx1[0]
        else:
            choice2 = random.random()
            if choice2 < 0.5:
                indx1 = gp.mutEphemeral(ind, "all")
                ind = indx1[0]
            else:
                indx1 = gp.mutEphemeral(ind, "one")
                ind = indx1[0]
    return ind,