from argparse import Namespace
import torch
from models.neural_epi_fitter import NeuralEpiFitter, load_fitter, segment_diagnostics_to_excel
from models.neural_epi_modules import build_neural_epi
from utils.general_utils import set_seed, make_run_dir, run_cell_dir, finish_run_dir, model_spec_tag, epi_para_tag
from utils.spec_args import get_args, build_spec, build_cfgs
from dataloader.location_data import load_data_record, tensors_from_record, scenario_provider, LOCATIONS
from utils.epi_base import load_epi_base


def resume_train_run(ns, data_rel_path, model_cfg, fit_cfg, train):
    train_dir = run_cell_dir(ns, data_rel_path, "train") / f"se_{ns.seed}"
    ckpt_path = train_dir / "final_fitter.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--mode test needs this cell's train run first, no {ckpt_path}.")

    fitter, extra = load_fitter(ckpt_path, fit_cfg=fit_cfg)

    train_points = train[0].shape[1]
    if extra.get("segment_end") != train_points:
        raise RuntimeError(f"{train_dir} is incomplete. Delete it and refit.")
    saved = dict(fitter.model_cfg.__dict__)
    if saved != vars(model_cfg):
        differs = [k for k in vars(model_cfg) if saved.get(k) != vars(model_cfg)[k]]
        raise RuntimeError(f"{ckpt_path} was fitted with a different model config: {differs}.")

    start = int(extra["segment_start"])
    init_state = extra["init_state"]
    if init_state is None:
        with torch.no_grad():
            *_, x0, _ = fitter.forward_segment(train[0][:, start:start + 2, :], None, with_beta=False)
        init_state = x0.detach().unsqueeze(0).contiguous()

    return fitter, start, init_state


def run_one_seed(spec, seed, args, data_df, data_rel_path, stats, epi):
    epi_version, epi_base = epi
    feats, model_cfg, fit_cfg = build_cfgs(spec, epi_base)
    set_seed(seed)

    ns = Namespace(
        beta_input_feats=feats,
        epi_tag=epi_para_tag(model_cfg, epi_base, epi_version),
        beta_model_hidden_dim=model_cfg.beta_model_hidden_dim,
        beta_pen_lam=fit_cfg.beta_pen_lam,
        segment_len=fit_cfg.segment_len,
        seed=seed,
    )

    train, full, dates = tensors_from_record(data_df, feats)

    if args.mode == "test":
        fitter, start, init_state = resume_train_run(ns, data_rel_path, model_cfg, fit_cfg, train)
        X, Y, POP = full[0][:, start:, :], full[1][:, start:], full[2][start:, :]
        span_dates, record_for_X = dates[start:], data_df.iloc[start:]
    else:
        start, init_state = 0, None
        X, Y, POP = train
        span_dates = dates[:X.shape[1]]
        record_for_X = data_df[data_df["split"] == "train"]
        fitter = NeuralEpiFitter(build_neural_epi(model_cfg), fit_cfg)

    run_dir = make_run_dir(ns, data_rel_path, mode=args.mode)
    diag_path = run_dir / "segment_diagnostics.xlsx"

    def write_diagnostics(segment_diagnostics):
        segment_diagnostics_to_excel(segment_diagnostics=segment_diagnostics, dates=span_dates,
                                     state_names=fitter.neural_epi.mechanism.state_names, file_path=diag_path,
                                     feats=feats)

    alt_covariates = {sc: scenario_provider(args.location, record_for_X, feats, sc, stats, args.alt_horizon)
                      for sc in args.alt_scenarios}

    fitter.fit(X=X, Y=Y, POP=POP,
               save_path=run_dir / "final_fitter.pt", data_rel_path=str(data_rel_path),
               feats=feats, alt_covariates=alt_covariates, dates=span_dates,
               on_segment=write_diagnostics, init_state=init_state)

    finish_run_dir(run_dir)
    print(f"Segment diagnostics saved to: {diag_path}")


def main():
    args = get_args()
    spec = build_spec(args)
    epi = load_epi_base(args.location, LOCATIONS)
    build_cfgs(spec, epi[1])

    data_df, data_rel_path, stats = load_data_record(args.location, args.test_steps, start_date=args.start_date)
    for seed in range(args.seed_start, args.seed_start + args.n_inits):
        run_one_seed(spec, seed, args, data_df, data_rel_path, stats, epi)


if __name__ == "__main__":
    main()
