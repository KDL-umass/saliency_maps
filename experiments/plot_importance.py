from saliency_maps.experiments import CONCEPTS, SAVE_DIR, INTERVENTIONS

import numpy as np
import argparse
import pickle
import random

import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

GAME = None

'''
Scatter plot of SM vs CF importance per time step in episode.
'''
def plot_impCorr_perFrame(episode_importance, num_samples, imp_type="action", saliency_method='perturbation'):
    global GAME

    #plot per time step
    for i,instance in enumerate(episode_importance):
        #save data in appropriate format per intervention
        SM_imp = []
        CF_imp = []
        for concept in CONCEPTS[GAME]:
            SM_imp += [instance[concept]["SM_imp"]]
            CF_imp += [np.mean(instance[concept]["CF_imp"], axis=0)]

        #plot
        plt.figure()
        for j, cf_concept_imp in enumerate(CF_imp):
            sm_concept_imp = len(cf_concept_imp)*[SM_imp[j]] #multiply SM importance by number of interventions
            plt.scatter(sm_concept_imp, cf_concept_imp, label=CONCEPTS[GAME][j])
            if imp_type=="action":
                plt.ylabel('Euclidean Distance of Network Action Logits')
            else:
                plt.ylabel('Euclidean Distance of Network Value')
            plt.xlabel('Saliency Score')
            plt.title('Saliency Importance VS Counterfactual Importance for Each Object')
            plt.legend()
        plt.savefig(SAVE_DIR + 'CF_imp/{}/default-150-breakouttoyboxnoframeskip-v4-56/num_samples_{}/frame{}_{}Imp.png'.format(saliency_method, num_samples, i+30, imp_type))

'''
Scatter plot of SM vs CF importance per intervention.
'''
def plot_impCorr_perIV(episode_importance, num_samples, imp_type="action", saliency_method='perturbation'):
    global GAME
    COLORS = {'perturbation': 'C0', 'object': 'red', 'jacobian': 'purple'}
    concepts = CONCEPTS[GAME]

    SM_imp = {}
    CF_imp = {}
    for concept in concepts: CF_imp[concept] = []
    for concept in concepts: SM_imp[concept] = []
    
    #save data in appropriate format per timestep
    for i,instance in enumerate(episode_importance):
        for concept in concepts:
            # interventions = INTERVENTIONS["bricks"] if "bricks" in concept else INTERVENTIONS[concept]
            SM_imp[concept] += [instance[concept]["SM_imp"]]
            CF_imp[concept] += [np.mean(instance[concept]["CF_imp"], axis=0)]

    #plot per intervention
    for concept in CF_imp.keys():
        interventions = INTERVENTIONS["bricks"] if "bricks" in concept else INTERVENTIONS[concept]
        CF_imp_concept = list(zip(*CF_imp[concept])) #separating by columns (ie. interventions)
        for i, IV in enumerate(interventions):
            plt.figure()
            plt.scatter(SM_imp[concept], CF_imp_concept[i], color=COLORS[saliency_method])
            plt.ylim(0, 30)
            # plt.yticks([])
            # if imp_type=="action":
            #     plt.ylabel('Euclidean Distance of Network Action Logits')
            # else:
            #     plt.ylabel('Euclidean Distance of Network Value')
            # plt.xlabel('Saliency Score')
            # plt.title('SM Importance VS CF Importance for {}'.format(IV))
            plt.savefig(SAVE_DIR + 'CF_imp/{}/default-150-breakouttoyboxnoframeskip-v4-56/num_samples_{}/IV{}Imp_{}.png'.format(saliency_method, num_samples, imp_type, IV))

'''
Line plot of importance over time.
'''
def plot_imp_overTime(episode_importance, num_samples, imp_type="action", saliency_method='perturbation'):
    global GAME
    concepts = CONCEPTS[GAME]

    SM_imp = {}
    CF_imp = {}
    for concept in concepts: CF_imp[concept] = []
    for concept in concepts: SM_imp[concept] = []
    
    #save data in appropriate format per timestep
    for i,instance in enumerate(episode_importance):
        for concept in concepts:
            SM_imp[concept] += [instance[concept]["SM_imp"]]
            CF_imp[concept] += [np.mean(instance[concept]["CF_imp"], axis=0)]

    #plot per intervention
    for concept in CF_imp.keys():
        interventions = INTERVENTIONS["bricks"] if "bricks" in concept else INTERVENTIONS[concept]
        CF_imp_concept = list(zip(*CF_imp[concept])) #separating by columns (ie. interventions)
        plt.figure()
        for i in np.argsort(SM_imp[concept])[100:]:
            plt.axvline(x=i, linestyle='-', color='black', alpha=0.2)
        # plt.plot(SM_imp[concept], label="SM_imp")
        # plt.fill_between(range(len(SM_imp[concept])), np.subtract([t[i] for t in distances], [s[i] for s in saliency]), np.add([t[i] for t in distances], [s[i] for s in saliency]), alpha=0.2)
        for i, IV in enumerate(interventions):
            plt.plot(CF_imp_concept[i], label="CF " + IV)
            plt.ylabel('{} Importance'.format(imp_type.upper()))
            plt.xlabel('Time')
            plt.title('Cummulative Importance Over an Episode')
            plt.legend()
        plt.savefig(SAVE_DIR + 'CF_imp/{}/default-150-breakouttoyboxnoframeskip-v4-56/num_samples_{}/{}Imp_{}.png'.format(saliency_method, num_samples, imp_type, concept))

'''
Box plot of CF importance per concept.
'''
def plot_CFimp_variability(episode_importance, num_samples, imp_type="action", saliency_method='perturbation'):
    global GAME
    concepts = CONCEPTS[GAME]

    SM_imp = {}
    CF_imp = {}
    for concept in concepts: CF_imp[concept] = []
    for concept in concepts: SM_imp[concept] = []
    
    #save data in appropriate format per timestep
    for i,instance in enumerate(episode_importance):
        for concept in concepts:
            SM_imp[concept] += [instance[concept]["SM_imp"]]
            temp_CF_imp = list(zip(*instance[concept]["CF_imp"])) #separating by columns (ie. interventions)
            for j in range(len(temp_CF_imp)):
                if i == 0:
                    CF_imp[concept] += [list(temp_CF_imp[j])]
                else:
                    CF_imp[concept][j] += list(temp_CF_imp[j])

    print("SM_imp: ", SM_imp)
    print("CF_imp: ", CF_imp)

    new_imp = []
    for concept in CF_imp.keys():
        print(len(CF_imp[concept]))
        new_imp += [CF_imp[concept]]

    print(new_imp)

    #plot per intervention
    plt.subplots()
    plt.boxplot(new_imp)
    plt.xticks([])
    plt.title('CF Importance Variability Over Episode')
    plt.ylabel('CF {} Importance'.format(imp_type.upper()))
    plt.xlabel('Concept')
    # for concept in CF_imp.keys():
    #     interventions = INTERVENTIONS["bricks"] if "bricks" in concept else INTERVENTIONS[concept]
    #     plt.boxplot(CF_imp[concept])
    #     plt.ylabel('CF Importance')
    #     plt.xlabel('Intervention Type')
    #     plt.title('CF Importance Variability Over Episode')
        # for i, IV in enumerate(interventions):
        #     # sm_concept_imp = list(np.repeat(SM_imp[concept], num_samples)) #multiply SM importance by number of samples
        #     plt.boxplot(CF_imp[concept][i])
        #     plt.ylabel('CF Importance')
        #     plt.xlabel('Intervention Type')
        #     plt.title('CF Importance Variability Over Episode')
        #     # plt.legend()
    plt.savefig(SAVE_DIR + 'CF_imp/{}/default-150-breakouttoyboxnoframeskip-v4-56/num_samples_{}/box_plt_{}{}.png'.format(saliency_method, num_samples, imp_type, concept))

'''
Scatter plot for CF importance versus intensity of intervention per intervention.
'''
def plot_IVintensity_corr(episode_importance, num_samples, imp_type="action", saliency_method='perturbation'):
    global GAME
    concepts = CONCEPTS[GAME]

    IV_intensity = {}
    CF_imp = {}
    for concept in concepts: CF_imp[concept] = []
    for concept in concepts: IV_intensity[concept] = []
    
    #save data in appropriate format per timestep
    for i,instance in enumerate(episode_importance):
        for concept in concepts:
            temp_CF_imp = list(zip(*instance[concept]["CF_imp"])) #separating by columns (ie. interventions)
            temp_intensity = list(zip(*instance[concept]["IV_intensity"]))
            for j in range(len(temp_CF_imp)):
                if i == 0:
                    CF_imp[concept] += [list(temp_CF_imp[j])]
                    IV_intensity[concept] += [list(temp_intensity[j])]
                else:
                    CF_imp[concept][j] += list(temp_CF_imp[j])
                    IV_intensity[concept][j] += list(temp_intensity[j])

    #plot per intervention
    for concept in CF_imp.keys():
        interventions = INTERVENTIONS["bricks"] if "bricks" in concept else INTERVENTIONS[concept]
        for i, IV in enumerate(interventions):
            plt.figure()
            plt.scatter(IV_intensity[concept][i], CF_imp[concept][i], color='green', alpha=0.1)
            plt.ylabel('CF Importance')
            plt.xlabel('Intervention Intensity')
            plt.title('Intervention Intensity VS CF Importance for {}'.format(IV))
            plt.savefig(SAVE_DIR + 'CF_imp/{}/default-150-breakouttoyboxnoframeskip-v4-56/num_samples_{}/IV_intensity_correlation_{}.png'.format(saliency_method, num_samples, IV))

def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-n', '--num_samples', default=10, type=int, help='number of samples to compute importance over')
    parser.add_argument('-g', '--game', default="Breakout", help='game we are computing the plots for')
    parser.add_argument('-s', '--saliency_method', default="perturbation", help='saliency method to be used')
    args = parser.parse_args()

    GAME = args.game
    load_path = './saliency_maps/experiments/results/CF_imp/{}/default-150-breakouttoyboxnoframeskip-v4-56/num_samples_{}'.format(args.saliency_method, args.num_samples)

    with open(load_path + "/episode_actionImp.pkl", 'rb') as f:
        episode_actionImp = pickle.load(f)
    with open(load_path + "/episode_valueImp.pkl", 'rb') as f:
        episode_valueImp = pickle.load(f)

    plot_impCorr_perFrame(episode_actionImp, args.num_samples, saliency_method=args.saliency_method)
    plot_impCorr_perIV(episode_actionImp, args.num_samples, saliency_method=args.saliency_method)
    plot_imp_overTime(episode_actionImp, args.num_samples, saliency_method=args.saliency_method)
    plot_CFimp_variability(episode_actionImp, args.num_samples, saliency_method=args.saliency_method)
    plot_IVintensity_corr(episode_actionImp, args.num_samples, saliency_method=args.saliency_method)

    plot_impCorr_perFrame(episode_valueImp, args.num_samples, imp_type="value", saliency_method=args.saliency_method)
    plot_impCorr_perIV(episode_valueImp, args.num_samples, imp_type="value", saliency_method=args.saliency_method)
    plot_imp_overTime(episode_valueImp, args.num_samples, imp_type="value", saliency_method=args.saliency_method)
    plot_CFimp_variability(episode_valueImp, args.num_samples, imp_type="value", saliency_method=args.saliency_method)
    plot_IVintensity_corr(episode_valueImp, args.num_samples, imp_type="value", saliency_method=args.saliency_method)
