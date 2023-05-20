from experiments.models import bloomz, flan_t5_large, flan_t5_xl, t0_3b, tk_instruct_3b_def, mt0


def get_language_models():
    return {
        "bloomz_1b1": lambda: bloomz("1b1"),
        "bloomz_1b7": lambda: bloomz("1b7"),
        "bloomz_3b": lambda: bloomz("3b"),
        "bloomz_560m": lambda: bloomz("560m"),
        "flan_t5_xl": flan_t5_xl,
        "flan_t5_large": flan_t5_large,
        "t0_3b": t0_3b,
        "tk_instruct_3b_def": tk_instruct_3b_def,
        "mt0_large": lambda: mt0("mt0-large"),
        "mt0_base": lambda: mt0("mt0-base"),
        "mt0_small": lambda: mt0("mt0-small"),
        "mt0_xl": lambda: mt0("mt0-xl"),
    }
