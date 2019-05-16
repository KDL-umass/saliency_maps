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
def plot_impCorr_perFrame(episode_importance, num_samples, imp_type="action"):
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
            plt.ylabel('Euclidean Distance of Network Action Logits')
            plt.xlabel('Saliency Score')
            plt.title('Saliency Importance VS Counterfactual Importance for Each Object')
            plt.legend()
        plt.savefig(SAVE_DIR + 'default-150-breakouttoyboxnoframeskip-v4-56/num_samples_{}/frame{}_{}Imp.png'.format(num_samples, i+30, imp_type))

'''
Scatter plot of SM vs CF importance per intervention.
'''
def plot_impCorr_perIV(episode_importance, num_samples, imp_type="action"):
    global GAME
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
            plt.scatter(SM_imp[concept], CF_imp_concept[i])
            plt.ylabel('Euclidean Distance of Network Action Logits')
            plt.xlabel('Saliency Score')
            plt.title('SM Importance VS CF Importance for {}'.format(IV))
            plt.savefig(SAVE_DIR + 'default-150-breakouttoyboxnoframeskip-v4-56/num_samples_{}/IV{}Imp_{}.png'.format(num_samples, imp_type, IV))

'''
Line plot of importance over time.
'''
def plot_imp_overTime(episode_importance, num_samples, imp_type="action"):
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
        plt.plot(SM_imp[concept], label="SM {}".format(concept))
        for i, IV in enumerate(interventions):
            plt.plot(CF_imp_concept[i], label="CF " + IV)
            plt.ylabel('Importance')
            plt.xlabel('Time')
            plt.title('Cummulative Importance Over an Episode')
            plt.legend()
        plt.savefig(SAVE_DIR + 'default-150-breakouttoyboxnoframeskip-v4-56/num_samples_{}/{}Imp_{}.png'.format(num_samples, imp_type, concept))

'''
Box plot of CF importance per intervention.
'''
def plot_CFimp_variability(episode_importance, num_samples, imp_type="action"):
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
    plt.savefig(SAVE_DIR + 'default-150-breakouttoyboxnoframeskip-v4-56/num_samples_{}/box_plt_{}{}.png'.format(num_samples, imp_type, concept))

'''
Scatter plot for CF importance versus intensity of intervention per intervention.
'''
def plot_IVintensity_corr(episode_importance, num_samples, imp_type="action"):
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
            plt.savefig(SAVE_DIR + 'default-150-breakouttoyboxnoframeskip-v4-56/num_samples_{}/IV_intensity_correlation_{}.png'.format(num_samples, IV))

def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--load_path', help='path to load the pickle file containing importance information from')
    parser.add_argument('-n', '--num_samples', default=10, type=int, help='number of samples to compute importance over')
    parser.add_argument('-g', '--game', default="Breakout", help='game we are computing the plots for')
    args = parser.parse_args()

    GAME = args.game

    with open(args.load_path + "/episode_actionImp.pkl", 'rb') as f:
        episode_actionImp = pickle.load(f)
    with open(args.load_path + "/episode_valueImp.pkl", 'rb') as f:
        episode_valueImp = pickle.load(f)

    plot_impCorr_perFrame(episode_actionImp, args.num_samples)
    plot_impCorr_perIV(episode_actionImp, args.num_samples)
    plot_imp_overTime(episode_actionImp, args.num_samples)
    plot_CFimp_variability(episode_actionImp, args.num_samples)
    plot_IVintensity_corr(episode_actionImp, args.num_samples)

    plot_impCorr_perFrame(episode_valueImp, args.num_samples, imp_type="value")
    plot_impCorr_perIV(episode_valueImp, args.num_samples, imp_type="value")
    plot_imp_overTime(episode_valueImp, args.num_samples, imp_type="value")
    plot_CFimp_variability(episode_valueImp, args.num_samples, imp_type="value")
    plot_IVintensity_corr(episode_valueImp, args.num_samples, imp_type="value")
