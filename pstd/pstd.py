"""
This module contains an implementation of the k-space PSTD method.

This implementation uses numexpr to accelerate the array manipulations.
"""
import numpy as np
import numexpr as ne

import logging

# logger = logging.getLogger(__name__)    # Use module name as logger name


try:
    from pyfftw.interfaces.numpy_fft import (
        fft2,
        ifft2,
    )  # Performs much better than numpy's fftpack
except ImportError:  # Use monkey-patching np.fft perhaps instead?
    from numpy.fft import fft2, ifft2

    logging.info("pyFFTW not available. Using NumPy FFT.")

from .model import Model


def kappa(wavenumber, timestep, c):
    r"""
    k-space operator.

    :param wavenumber: Wavenumber :math:`k`
    :param timestep: Timestep :math:`\Delta t`

    .. math:: \kappa = \mathrm{sinc}{\left( c_0 \Delta t k / 2 \right)}

    """
    # return ne.evaluate("sin(c * timestep * wavenumber / 2.0) / (c * timestep * wavenumber / 2.0)")
    return np.sinc(c * timestep * wavenumber / 2.0)


def to_pressure_gradient(pressure_fft, wavenumber, kappa, spacing):
    r"""
    Pressure gradient.

    :param wavenumber: Wavenumber :math:`k`
    :param kappa: k-space operator :math:`\kappa`
    :param spacing: Spacing :math:`\Delta \xi`
    :param pressure: Pressure at timestep :math:`p^n`

    .. math:: \frac{\partial }{\partial \xi} p^n = \mathcal{F}^{-1} \{ i k_{\xi} \kappa e^{i k_{\xi} \Delta \xi / 2 } \mathcal{F} \{ p^n \} \}

    K-space documentation Equation 2.17a as well as
    """
    # return ifft(+1j * wavenumber * kappa * np.exp(+1j*wavenumber*spacing/2.0) * fft(pressure, axis=axis), axis=axis)
    j = 1j
    return ne.evaluate(
        "+j * wavenumber * kappa * exp(+j * wavenumber*spacing/2.0) * pressure_fft"
    )
    # return (+1j * wavenumber * kappa * np.exp(+1j*wavenumber*spacing/2.0) * pressure_fft)#fft2(pressure))


def to_velocity_gradient(velocity_fft, wavenumber, kappa, spacing):
    r"""
    Velocity gradient.

    :param wavenumber: Wavenumber :math:`k`
    :param kappa: k-space operator :math:`\kappa`
    :param spacing: Spacing :math:`\Delta \xi`
    :param velocity: Pressure at timestep :math:`u_{\xi}^{n+\frac{1}{2}}`

    .. math:: \frac{\partial }{\partial \xi} u_{\xi}^{n+\frac{1}{2}} = \mathcal{F}^{-1} \{i k_{\xi} \kappa e^{-i k_{\xi} \Delta \xi / 2 } \mathcal{F} \{ u_{\xi}^{n+\frac{1}{2}} \} \}

    Equation 2.17c.
    """
    # return ifft(+1j * wavenumber * kappa * np.exp(-1j*wavenumber*spacing/2.0) * fft(velocity, axis=axis), axis=axis)
    j = 1j
    return ne.evaluate(
        "+j * wavenumber * kappa * exp(-j * wavenumber*spacing/2.0) * velocity_fft"
    )
    # return (+1j * wavenumber * kappa * np.exp(-1j*wavenumber*spacing/2.0) * velocity_fft)


def abs_exp(alpha, timestep):
    r"""
    Absorption coefficient exponent.

    :param alpha: :math:`\alpha_{\xi}`
    :param timestep: Timestep :math:`\Delta t`

    This value is calculated according to

    .. math:: e^{-\alpha_{\xi} \Delta t / 2}

    """
    return ne.evaluate("exp(-alpha * timestep / 2.0)")
    # return np.exp(alpha * -timestep / 2.0)


def pressure_abs_exp(alpha, timestep):
    r"""
    Absorption coefficient exponent.

    :param alpha: :math:`\alpha_{\xi}`
    :param timestep: Timestep :math:`\Delta t`

    This value is calculated according to

    .. math:: e^{-\alpha_{\xi} \Delta t / 2}

    """
    return ne.evaluate("exp(-alpha * timestep / 2.0)")
    # return np.exp(alpha * -timestep / 2.0)


def velocity_abs_exp(alpha, timestep, spacing, wavenumber):
    r"""
    Absorption coefficient exponent.

    :param alpha: :math:`\alpha_{\xi}`
    :param timestep: Timestep :math:`\Delta t`

    This value is calculated according to

    .. math:: e^{-\alpha_{\xi} \Delta t / 2}

    However, since the velocity field is shifted by half a spacing, a correction needs to be applied.

    .. math:: \mathcal{F}^{-1} \left[ e^{+j k_{\xi} \Delta \xi / 2} \mathcal{F} \left( e^{-\alpha_{\xi} \Delta t / 2} \right) \right]

    """
    j = 1j
    return ifft2(
        ne.evaluate("exp(+j * wavenumber*spacing/2.0)")
        * fft2(ne.evaluate("exp(-alpha * timestep / 2.0)"))
    )
    # return ifft2(np.exp(+1j*wavenumber*spacing/2.0) * fft2(np.exp(alpha * -timestep / 2.0)) )


def velocity_with_pml(
    previous_velocity, pressure_gradient, timestep, density, abs_exp, source
):
    r"""
    Velocity.

    :param previous_velocity: Velocity at previous timestep.
    :param pressure_gradient:  Pressure gradient at previous timestep.
    :param timestep: Timestep :math:`\Delta t`
    :param density: Density :math:`\rho_0`
    :param abs_exp: Absorption exponent :math:`e^{-\alpha_{\xi} \Delta t / 2}`
    :param source: Source term :math:`S^n_{F_{\xi}}`


    .. math:: u_{\xi}^{n+\frac{1}{2}} = e^{-\alpha_{\xi} \Delta t / 2} \left( e^{-\alpha_{\xi} \Delta t / 2} u_{\xi}^{n-\frac{1}{2}} - \frac{\Delta t}{\rho_0} \frac{\partial}{\partial \xi} p^n + \Delta t S^n_{F_{\xi}} \right)

    Equation 2.27.
    """
    return ne.evaluate(
        "abs_exp * (abs_exp * previous_velocity - timestep / density * pressure_gradient  + timestep * source)"
    )
    # return abs_exp * (abs_exp * previous_velocity - timestep / density * pressure_gradient  + timestep * source)


def pressure_with_pml(
    previous_pressure, velocity_gradient, timestep, density, soundspeed, abs_exp, source
):
    r"""
    Pressure.

    :param previous_pressure: Pressure at previous timestep.
    :param velocity_gradient: Velocity gradient at previous timestep.
    :param timestep: Timestep :math:`\Delta t`
    :param density: Density :math:`\rho_0`
    :param soundspeed: Speed of sound :math:`c`.
    :param abs_exp: Absorption exponent :math:`e^{-\alpha_{\xi} \Delta t / 2}`
    :param source: Source term :math:`S^n_{M_{\xi}}`.

    .. math:: p_{\xi}^{n+1} = e^{-\alpha_{\xi} \Delta t / 2} \left( e^{-\alpha_{\xi} \Delta t / 2} u_{\xi}^{n} - \Delta t \rho_0 c^2 \frac{\partial}{\partial \xi} v^n + \Delta t S^n_{M_{\xi}} \right)

    """
    return ne.evaluate(
        "abs_exp * (abs_exp * previous_pressure - timestep * (density * soundspeed**2.0)  * velocity_gradient + timestep * source)"
    )
    # return abs_exp * (abs_exp * previous_pressure - timestep * (density * soundspeed**2.0)  * velocity_gradient + timestep * source)


def sync_steps(
    p,
    v,
    p_fft,
    k,
    kappa,
    spacing,
    timestep,
    density,
    soundspeed,
    abs_exp_p,
    abs_exp_v,
    source_p,
    source_v,
):

    v = velocity_with_pml(
        v,
        ifft2(to_pressure_gradient(p_fft, k, kappa, spacing)),
        timestep,
        density,
        abs_exp_v,
        source_v,
    )
    p = pressure_with_pml(
        p,
        ifft2(to_velocity_gradient(fft2(v), k, kappa, spacing)),
        timestep,
        density,
        soundspeed,
        abs_exp_p,
        source_p,
    )

    return p, v


# @classmethod
def update(d):
    r"""
    Calculation steps to be taken every step.

    :param d: Dictionary containing simulation data.

    .. note:: This method should only contain calculation steps.

    """
    step = d["step"]

    # Calculate FFT of pressure
    pressure_fft = fft2(
        d["field"][("pressure", None)]
    )  # Apply atmospheric absorption here?

    # out_x = d['executor'].submit(sync_steps, d['temp']['p_x'], d['field'][('velocity', 'x')], pressure_fft, d['k_x'],
    # d['kappa'], d['spacing'], d['timestep'], d['density'], d['soundspeed'],
    # d['abs_exp']['p']['x'], d['abs_exp']['v']['x'], d['source'][('pressure', None)], d['source'][('velocity', 'x')])

    # out_y = d['executor'].submit(sync_steps, d['temp']['p_y'], d['field'][('velocity', 'y')], pressure_fft, d['k_y'],
    # d['kappa'], d['spacing'], d['timestep'], d['density'], d['soundspeed'],
    # d['abs_exp']['p']['y'], d['abs_exp']['v']['y'], d['source'][('pressure', None)], d['source'][('velocity', 'y')])

    # out_x = out_x.result()
    # out_y = out_y.result()

    out_x = sync_steps(
        d["temp"]["p_x"],
        d["field"][("velocity", "x")],
        pressure_fft,
        d["k_x"],
        d["kappa"],
        d["spacing"],
        d["timestep"],
        d["density"],
        d["soundspeed"],
        d["abs_exp"]["p"]["x"],
        d["abs_exp"]["v"]["x"],
        d["source"][("pressure", None)],
        d["source"][("velocity", "x")],
    )

    out_y = sync_steps(
        d["temp"]["p_y"],
        d["field"][("velocity", "y")],
        pressure_fft,
        d["k_y"],
        d["kappa"],
        d["spacing"],
        d["timestep"],
        d["density"],
        d["soundspeed"],
        d["abs_exp"]["p"]["y"],
        d["abs_exp"]["v"]["y"],
        d["source"][("pressure", None)],
        d["source"][("velocity", "y")],
    )

    d["temp"]["p_x"], d["field"][("velocity", "x")] = out_x
    d["temp"]["p_y"], d["field"][("velocity", "y")] = out_y

    d["field"][("pressure", None)] = d["temp"]["p_x"] + d["temp"]["p_y"]

    return d


class PSTD(Model):
    r"""
    K-space Pseudo Spectral Time-Domain model.
    """

    # ('velocity', ('x', 'y'))]

    _update = staticmethod(update)

    ##fft_wisdom_forward = None
    ##"""FFTW wisdom file for forward transform.
    ##"""
    ##fft_wisdom_backward = None
    ##"""FFTW wisdom file for backward transform.
    ##"""

    @staticmethod
    def stability_criterion(CFL, c_0, c_ref):
        r"""
        K-space PSTD stability criterium as function of CFL.

        :param CFL: CFL
        :param c_0: Speed of sound field values :math:`c_0`
        :param c_ref: Reference speed of sound :math:`c_{ref}`

        .. math:: CFL <= \frac{2}{\pi} \frac{c_0}{c_{ref}} \sin^{-1}{\left(\frac{c_{ref}}{c_{0}} \right)}

        """
        return CFL <= 2.0 / np.pi * (c_0 / c_ref) * np.arcsin(c_ref / c_0)

    # def _pre_run(self, data):

    # super()._pre_run(data)
    # data['executor'] = ProcessPoolExecutor(max_workers=2)
    # return data

    # def _post_run(self, data):
    # data['executor'].shutdown()
    # super()._post_run(data)
    # return data

    def _pre_start(self, data):

        data = super()._pre_start(data)

        # print (data)

        data["k_x"], data["k_y"] = np.meshgrid(
            self.axes.x.wavenumbers, self.axes.y.wavenumbers, indexing="ij"
        )
        data["k_x"] = data["k_x"].astype(self.dtype("float"))
        data["k_y"] = data["k_y"].astype(self.dtype("float"))

        data["k"] = np.sqrt(data["k_x"] ** 2.0 + data["k_y"] ** 2.0)

        data["kappa"] = kappa(
            data["k"], data["timestep"], np.mean(self.medium.soundspeed)
        )  # Independent of time when considering a frozen field.
        data["density"] = self.medium.density * np.ones(
            self.grid.shape, dtype=self.dtype("float")
        )
        data["soundspeed"] = (
            self.medium.soundspeed_for_calculation * np.ones(self.grid.shape)
        ).astype(self.dtype("float"))

        data["abs_exp"] = {"p": dict(), "v": dict()}
        data["abs_exp"]["p"]["x"] = pressure_abs_exp(
            data["pml"]["x"], data["timestep"]
        )  # Absorption exponent for x-direction
        data["abs_exp"]["p"]["y"] = pressure_abs_exp(
            data["pml"]["y"], data["timestep"]
        )  # Absorption exponent for y-direction
        data["abs_exp"]["v"]["x"] = velocity_abs_exp(
            data["pml"]["x"], data["timestep"], data["spacing"], data["k_x"]
        )  # Absorption exponent for x-direction
        data["abs_exp"]["v"]["y"] = velocity_abs_exp(
            data["pml"]["y"], data["timestep"], data["spacing"], data["k_y"]
        )  # Absorption exponent for y-direction

        data["size"] = self.grid.size
        data["shape"] = self.grid.shape

        data["temp"] = dict()
        data["temp"]["p_x"] = np.zeros_like(data["field"][("pressure", None)])
        data["temp"]["p_y"] = np.zeros_like(data["field"][("pressure", None)])

        return data
