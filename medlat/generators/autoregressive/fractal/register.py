from medlat.registry import register_model
from .models import (
    FractalAR_in64,
    FractalMAR_in64,
    FractalMAR_base_in256,
    FractalMAR_large_in256,
    FractalMAR_huge_in256,
)

__all__ = []


@register_model("fractal.ar_64")
def fractal_ar_64(**kwargs):
    return FractalAR_in64(**kwargs)


@register_model("fractal.mar_64")
def fractal_mar_64(**kwargs):
    return FractalMAR_in64(**kwargs)


@register_model("fractal.mar_base_256")
def fractal_mar_base_256(**kwargs):
    return FractalMAR_base_in256(**kwargs)


@register_model("fractal.mar_large_256")
def fractal_mar_large_256(**kwargs):
    return FractalMAR_large_in256(**kwargs)


@register_model("fractal.mar_huge_256")
def fractal_mar_huge_256(**kwargs):
    return FractalMAR_huge_in256(**kwargs)
