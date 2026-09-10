import argparse
from model_analysis.ensemble import export_from_root, N_SEEDS


def get_args():
    parser = argparse.ArgumentParser(description="Export seed ensembles of one run cell to xlsx")
    parser.add_argument("--run_root", required=True, type=str,
                        help="checkpoints_local/<data_tag>/<epi_tag>/<spec_tag>")
    parser.add_argument("--alias", required=True, type=str, help="Output folder name under neural_epi_results/")
    parser.add_argument("--seed_start", default=0, type=int)
    parser.add_argument("--n_seeds", default=N_SEEDS, type=int, help="how many seed runs to load in total")
    parser.add_argument("--block", default=N_SEEDS, type=int, help="Seeds per ensemble.")
    parser.add_argument("--mode", default="test", type=str, choices=["train", "test"])
    return parser.parse_args()


def main():
    args = get_args()
    out_dir = export_from_root(args.run_root, args.alias, args.n_seeds, args.block, args.seed_start, args.mode)
    print(f"Ensembles saved to: {out_dir}")


if __name__ == "__main__":
    main()