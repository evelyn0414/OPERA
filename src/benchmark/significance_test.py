import pandas as pd
from scipy.stats import ttest_ind
from linear_eval import *

def get_performance(model, dim, args):
    feature = model
    if feature not in ["vggish", "opensmile", "clap",  "audiomae"]: # baselines
        feature += str(dim)

    if not args.LOOCV:
        # report mean and std for 5 runs with random seeds
        auc_scores = []
        for seed in range(args.n_run):

            # fix seeds for reproducibility
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            if args.task == "covid19sounds":
                auc = linear_evaluation_covid19sounds(1, feature, modality=args.modality, l2_strength=args.l2_strength, lr=args.lr, head=args.head)
            elif args.task == "icbhidisease":
                auc = linear_evaluation_icbhidisease(use_feature=feature, epochs=64, batch_size=32, l2_strength=args.l2_strength, lr=args.lr, head=args.head)
            elif args.task == "kauh":
                auc = linear_evaluation_kauh(use_feature=feature, epochs=50, batch_size=32, l2_strength=args.l2_strength, lr=args.lr, head=args.head)
            elif args.task == "coswarasmoker":
                auc = linear_evaluation_coswara(use_feature=feature, epochs=64, l2_strength=args.l2_strength, batch_size=32, lr=args.lr, modality=args.modality, label="smoker", head=args.head)
            elif args.task == "coswaracovid":
                auc = linear_evaluation_coswara(use_feature=feature, epochs=64, l2_strength=args.l2_strength, batch_size=32, lr=args.lr, modality=args.modality, label="covid", head=args.head)
            elif args.task == "coswarasex":
                auc = linear_evaluation_coswara(use_feature=feature, epochs=64, l2_strength=args.l2_strength, batch_size=32, lr=args.lr, modality=args.modality, label="sex", head=args.head)
            elif args.task == "copd":
                auc = linear_evaluation_copd(use_feature=feature, l2_strength=args.l2_strength, lr=args.lr, head=args.head, epochs=64)  
            elif args.task == "coughvidcovid":
                auc = linear_evaluation_coughvid(use_feature=feature, epochs=64, l2_strength=args.l2_strength, lr=args.lr, batch_size=64, label="covid", head=args.head)
            elif args.task == "coughvidgender":
                auc = linear_evaluation_coughvid(use_feature=feature, epochs=64, l2_strength=args.l2_strength, lr=args.lr, batch_size=64, label="gender", head=args.head)
            elif args.task == "coviduk":
                auc = linear_evaluation_coviduk(use_feature=feature, epochs=64, l2_strength=args.l2_strength, lr=args.lr, batch_size=64, modality=args.modality, head=args.head)
            elif args.task == "snoring":
                auc = linear_evaluation_ssbpr(use_feature=feature, l2_strength=args.l2_strength, lr=args.lr, head=args.head, epochs=32, seed=seed)
            auc_scores.append(auc)
        print("=" * 48)
        print(auc_scores)
        print("Five times mean task {} feature {} results: auc mean {:.3f} ± {:.3f}".format(args.task, feature, np.mean(auc_scores), np.std(auc_scores)) )
        print("=" * 48)
        return auc_scores
    else:
        # Leave one out cross validation
        
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        
        if args.task == "spirometry":
            maes, mapes = linear_evaluation_mmlung(use_feature=feature, method='LOOCV', l2_strength=1e-1, epochs=64, lr=1e-1, batch_size=64, modality=args.modality, label=args.label, head=args.head)  
        
        if args.task == "rr":
            maes, mapes = linear_evaluation_nosemic(use_feature=feature, method='LOOCV', l2_strength=1e-1, epochs=64, batch_size=64, lr=1e-4, head=args.head)

        print("=" * 48)
        print(maes)
        print(mapes)
        print("Five times mean task {} feature {} results: MAE mean {:.3f} ± {:.3f}".format(args.task, feature, np.mean(maes), np.std(maes)) )
        print("Five times mean task {} feature {} results: MAPE mean {:.3f} ± {:.3f}".format(args.task, feature, np.mean(mapes), np.std(mapes)) )
        print("=" * 48)
        return maes


def test_2models(args):
    alpha = args.alpha
    performance_1 = get_performance(args.model1, args.dim1, args)
    performance_2 = get_performance(args.model2, args.dim2, args)
    t_stat, p_val = ttest_ind(performance_1, performance_2)
    if p_val > alpha:
        print(f"> {alpha} ", "Fail to reject null hypotesis")
    else:
        print(f"<= {alpha} ", "Reject null hypotesis")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()

    # these args need to be entered according to tasks
    parser.add_argument("--task", type=str, default="covid19sounds")
    parser.add_argument("--label", type=str, default='smoker') # prediction target
    parser.add_argument("--modality", type=str, default="cough")
    parser.add_argument("--model1", type=str, default="operaCT")
    parser.add_argument("--model2", type=str, default="audiomae")
    parser.add_argument("--dim1", type=int, default=768) 
    parser.add_argument("--dim2", type=int, default=768) 
    parser.add_argument("--LOOCV", type=bool, default=False)
    parser.add_argument("--alpha", type=float, default=0.01)

    # these can follow default
    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--l2_strength", type=float, default=1e-5)
    parser.add_argument("--head", type=str, default="linear")
    parser.add_argument("--mapgoogle", type=bool, default=False) # align test set with HeAR
    parser.add_argument("--n_run", type=int, default=5)

    args = parser.parse_args()
    test_2models(args)

    
