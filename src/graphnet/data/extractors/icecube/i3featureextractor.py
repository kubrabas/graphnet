"""I3Extractor class(es) for extracting specific, reconstructed features."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional
import numpy as np
from .i3extractor import I3Extractor
from graphnet.data.extractors.icecube.utilities.frames import (
    get_om_keys_and_pulseseries,
)
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray, dataio  # pyright: reportMissingImports=false


class I3FeatureExtractor(I3Extractor):
    """Base class for extracting specific, reconstructed features."""

    def __init__(self, pulsemap: str, exclude: list = [None]):
        """Construct I3FeatureExtractor.

        Args:
            pulsemap: Name of the pulse (series) map for which to extract
                reconstructed features.
            exclude: List of keys to exclude from the extracted data.
        """
        # Member variable(s)
        self._pulsemap = pulsemap

        # Base class constructor
        super().__init__(pulsemap, exclude=exclude)


class I3FeatureExtractorPONE(I3Extractor):
    """Class for extracting reconstructed features for P-ONE events created with LeptonInjector."""

    def __init__(
        self,
        pulsemap: str,
        name: str = "feature",
        exclude: list = [None],
    ):
        self._pulsemap = pulsemap

        # Base class constructor
        super().__init__(extractor_name=name, exclude=exclude)
        self._extractor_name = name


        # are these angles correct?
        self._PMT_ANGLES = np.array(
            [
                [57.5, 270.],   # PMT 1
                [57.5, 0.],     # PMT 2
                [57.5, 90.],    # PMT 3
                [57.5, 180.],   # PMT 4
                [25., 225.],    # PMT 5
                [25., 315.],    # PMT 6
                [25., 45.],     # PMT 7
                [25., 135.],    # PMT 8
                [-57.5, 270.],  # PMT 9
                [-57.5, 180.],  # PMT 10
                [-57.5, 90.],   # PMT 11
                [-57.5, 0.],    # PMT 12
                [-25., 315.],   # PMT 13
                [-25., 225.],   # PMT 14
                [-25., 135.],   # PMT 15
                [-25., 45.],    # PMT 16
            ]
        )

        ## check this as well.
        self.MODULE_RADIUS_M = 0.2159

        self._pmt_x_coordinates_wrt_om_rotated = np.multiply(
            np.sin(np.deg2rad(90.0 - self._PMT_ANGLES[:, 0])),
            np.cos(np.deg2rad(self._PMT_ANGLES[:, 1])),
        )
        self._pmt_y_coordinates_wrt_om_rotated = np.multiply(
            np.sin(np.deg2rad(90.0 - self._PMT_ANGLES[:, 0])),
            np.sin(np.deg2rad(self._PMT_ANGLES[:, 1])),
        )
        self._pmt_z_coordinates_wrt_om_rotated = np.cos(
            np.deg2rad(90.0 - self._PMT_ANGLES[:, 0])
        )

        self._PMT_MATRIX_rotated = np.array(
            [
                self._pmt_x_coordinates_wrt_om_rotated,
                self._pmt_y_coordinates_wrt_om_rotated,
                self._pmt_z_coordinates_wrt_om_rotated,
            ]
        ).T

        self._PMT_COORDINATES_rotated = (
            self._PMT_MATRIX_rotated * self.MODULE_RADIUS_M
        )

        self._minus_90_degree_rotation_around_x_axis = np.array(
            [
                [1.000000e00, 0.000000e00, 0.000000e00],
                [0.000000e00, 0.000000e00, 1.000000e00],
                [0.000000e00, -1.000000e00, 0.000000e00],
            ],
            dtype=float,
        )

        self._PMT_COORDINATES_ORIGINAL = (
            self._PMT_COORDINATES_rotated
            @ self._minus_90_degree_rotation_around_x_axis.T
        )

    def set_gcd(self, gcd_file: Optional[str] = None) -> None:
        """Extract GFrame from gcd-file.


        Args:
            gcd_file: Path to GCD file. Defaults to None. 
        """

        gcd = dataio.I3File(gcd_file)


        # Get GFrame
        try:
            g_frame = gcd.pop_frame(icetray.I3Frame.Geometry)
            # If the line above fails, it means that no gcd file was given
        except RuntimeError as e:
            self.error(
                "No GCD file was provided "
            )
            raise e

        
        # Save information as member variables of I3Extractor
        self._gcd_dict = g_frame["I3Geometry"].omgeo
        if gcd_file is not None:
            self._gcd_file = gcd_file


    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, List[Any]]:
        """Extract reconstructed features from `frame`."""
        padding_value: float = -1.0
        output: Dict[str, List[Any]] = {
            "charge": [],
            "dom_time": [],
            "width": [],
            "dom_x": [],
            "dom_y": [],
            "dom_z": [],
            "pmt_area": [],
            "event_time": [],
            "string": [],
            "pmt_number": [],
            "dom_number": [],
            "dom_type": [],
            "pmt_x": [],
            "pmt_y": [],
            "pmt_z": [],
        }

        # Get OM data
        if self._pulsemap in frame:
            om_keys, data = get_om_keys_and_pulseseries(
                frame,
                self._pulsemap,
                self._calibration,
            )
        else:
            self.warning_once(f"Pulsemap {self._pulsemap} not found in frame.")
            return output



        
        event_time = frame["I3EventHeader"].start_time.mod_julian_day_double

        for om_key in om_keys:
            x = self._gcd_dict[om_key].position.x
            y = self._gcd_dict[om_key].position.y
            z = self._gcd_dict[om_key].position.z
            area = self._gcd_dict[om_key].area
           

            string = om_key[0]
            dom_number = om_key[1]
            pmt_number = om_key[2]
            dom_type = self._gcd_dict[om_key].omtype

            pmt_x = pmt_y = pmt_z = padding_value
            if pmt_number is not None:
                idx = int(pmt_number) - 1
                if 0 <= idx < len(self._PMT_COORDINATES_ORIGINAL):
                    rel = self._PMT_COORDINATES_ORIGINAL[idx]
                    pmt_x = x + float(rel[0])
                    pmt_y = y + float(rel[1])
                    pmt_z = z + float(rel[2])

  

            pulses = data[om_key]
            for pulse in pulses:
                output["charge"].append(
                    getattr(pulse, "charge", padding_value)
                )
                output["dom_time"].append(
                    getattr(pulse, "time", padding_value)
                )
                output["width"].append(getattr(pulse, "width", padding_value))
                output["pmt_area"].append(area)
                output["dom_x"].append(x)
                output["dom_y"].append(y)
                output["dom_z"].append(z)
                output["pmt_x"].append(pmt_x)
                output["pmt_y"].append(pmt_y)
                output["pmt_z"].append(pmt_z)

                output["string"].append(string)
                output["pmt_number"].append(pmt_number)
                output["dom_number"].append(dom_number)
                output["dom_type"].append(dom_type)

                output["event_time"].append(event_time)

              
        return output

    

class I3FeatureExtractorIceCube86(I3FeatureExtractor):
    """Class for extracting reconstructed features for IceCube-86."""

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, List[Any]]:
        """Extract reconstructed features from `frame`.

        Args:
            frame: Physics (P) I3-frame from which to extract reconstructed
                features.

        Returns:
            Dictionary of reconstructed features for all pulses in `pulsemap`,
                in pure-python format.
        """
        padding_value: float = -1.0
        output: Dict[str, List[Any]] = {
            "charge": [],
            "dom_time": [],
            "width": [],
            "dom_x": [],
            "dom_y": [],
            "dom_z": [],
            "pmt_area": [],
            "rde": [],
            "is_bright_dom": [],
            "is_bad_dom": [],
            "is_saturated_dom": [],
            "is_errata_dom": [],
            "event_time": [],
            "hlc": [],
            "awtd": [],
            "string": [],
            "pmt_number": [],
            "dom_number": [],
            "dom_type": [],
        }
        # Get OM data
        if self._pulsemap in frame:
            om_keys, data = get_om_keys_and_pulseseries(
                frame,
                self._pulsemap,
                self._calibration,
            )
        else:
            self.warning_once(f"Pulsemap {self._pulsemap} not found in frame.")
            return output

        # Added these :
        bright_doms = None
        bad_doms = None
        saturation_windows = None
        calibration_errata = None
        if "BrightDOMs" in frame:
            bright_doms = frame.Get("BrightDOMs")

        if "BadDomsList" in frame:
            bad_doms = frame.Get("BadDomsList")

        if "SaturationWindows" in frame:
            saturation_windows = frame.Get("SaturationWindows")

        if "CalibrationErrata" in frame:
            calibration_errata = frame.Get("CalibrationErrata")

        event_time = frame["I3EventHeader"].start_time.mod_julian_day_double

        for om_key in om_keys:
            # Common values for each OM
            x = self._gcd_dict[om_key].position.x
            y = self._gcd_dict[om_key].position.y
            z = self._gcd_dict[om_key].position.z
            area = self._gcd_dict[om_key].area
            rde = self._get_relative_dom_efficiency(
                frame, om_key, padding_value
            )

            string = om_key[0]
            dom_number = om_key[1]
            pmt_number = om_key[2]
            dom_type = self._gcd_dict[om_key].omtype

            # DOM flags
            if bright_doms:
                is_bright_dom = 1 if om_key in bright_doms else 0
            else:
                is_bright_dom = int(padding_value)

            if bad_doms:
                is_bad_dom = 1 if om_key in bad_doms else 0
            else:
                is_bad_dom = int(padding_value)

            if saturation_windows:
                is_saturated_dom = 1 if om_key in saturation_windows else 0
            else:
                is_saturated_dom = int(padding_value)

            if calibration_errata:
                is_errata_dom = 1 if om_key in calibration_errata else 0
            else:
                is_errata_dom = int(padding_value)

            # Loop over pulses for each OM
            pulses = data[om_key]
            for pulse in pulses:
                output["charge"].append(
                    getattr(pulse, "charge", padding_value)
                )
                output["dom_time"].append(
                    getattr(pulse, "time", padding_value)
                )
                output["width"].append(getattr(pulse, "width", padding_value))
                output["pmt_area"].append(area)
                output["rde"].append(rde)
                output["dom_x"].append(x)
                output["dom_y"].append(y)
                output["dom_z"].append(z)
                # ID's
                output["string"].append(string)
                output["pmt_number"].append(pmt_number)
                output["dom_number"].append(dom_number)
                output["dom_type"].append(dom_type)
                # DOM flags
                output["is_bright_dom"].append(is_bright_dom)
                output["is_bad_dom"].append(is_bad_dom)
                output["is_saturated_dom"].append(is_saturated_dom)
                output["is_errata_dom"].append(is_errata_dom)
                output["event_time"].append(event_time)

                # Pulse flags
                flags = getattr(pulse, "flags", padding_value)
                if flags == padding_value:
                    output["hlc"].append(padding_value)
                    output["awtd"].append(padding_value)
                else:
                    output["hlc"].append((pulse.flags >> 0) & 0x1)  # bit 0
                    output["awtd"].append(self._parse_awtd_flag(pulse))

        return output

    def _get_relative_dom_efficiency(
        self, frame: "icetray.I3Frame", om_key: int, padding_value: float
    ) -> float:
        if (
            "I3Calibration" in frame
        ):  # Not available for e.g. mDOMs in IceCube Upgrade
            rde = frame["I3Calibration"].dom_cal[om_key].relative_dom_eff
        else:
            try:
                assert self._calibration is not None
                rde = self._calibration.dom_cal[om_key].relative_dom_eff
            except:  # noqa: E722
                rde = padding_value
        return rde

    def _parse_awtd_flag(
        self, pulse: Any, fadc_min_width_ns: float = 6.0
    ) -> bool:
        """Parse awtd flag from pulse width.

        Returns True if the pulse was readout using the awtd digitizer.

        Method by Tom Stuttard.

        Notes from Tom:
        Function to read the bits of the pulse flags and unpack them into
        meaningful info Using pulse width rather than flags to separate FADC vs
        ATWD pulses, due to a known issue.
        https://github.com/icecube/icetray/issues/2721 Note that the issue
        states to use 8ns, but I have found that actually 6ns is correct.
        """
        # Use pulse width to check whether a pulse is
        # (a) FADC-only, or
        # includes ATWD (and probably also FADC)
        return pulse.width < (fadc_min_width_ns * icetray.I3Units.ns)


class I3FeatureExtractorIceCubeDeepCore(I3FeatureExtractorIceCube86):
    """Class for extracting reconstructed features for IceCube-DeepCore."""


class I3FeatureExtractorIceCubeUpgrade(I3FeatureExtractorIceCube86):
    """Class for extracting reconstructed features for IceCube-Upgrade."""

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, List[Any]]:
        """Extract reconstructed features from `frame`.

        Args:
            frame: Physics (P) I3-frame from which to extract reconstructed
                features.

        Returns:
            Dictionary of reconstructed features for all pulses in `pulsemap`,
                in pure-python format.
        """
        output: Dict[str, List[Any]] = {
            "pmt_dir_x": [],
            "pmt_dir_y": [],
            "pmt_dir_z": [],
        }

        # Add features from IceCube86
        output_icecube86 = super().__call__(frame)
        output.update(output_icecube86)

        # Get OM data
        if self._pulsemap in frame:
            om_keys, data = get_om_keys_and_pulseseries(
                frame,
                self._pulsemap,
                self._calibration,
            )
        else:
            self.warning_once(f"Pulsemap {self._pulsemap} not found in frame.")
            return output

        for om_key in om_keys:
            # Common values for each OM
            pmt_dir_x = self._gcd_dict[om_key].orientation.x
            pmt_dir_y = self._gcd_dict[om_key].orientation.y
            pmt_dir_z = self._gcd_dict[om_key].orientation.z

            # Loop over pulses for each OM
            pulses = data[om_key]
            for _ in pulses:
                output["pmt_dir_x"].append(pmt_dir_x)
                output["pmt_dir_y"].append(pmt_dir_y)
                output["pmt_dir_z"].append(pmt_dir_z)

        return output


class I3PulseNoiseTruthFlagIceCubeUpgrade(I3FeatureExtractorIceCube86):
    """Feature extractor class with pulse noise truth flag added."""

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, List[Any]]:
        """Extract reconstructed features from `frame`.

        Args:
            frame: Physics (P) I3-frame from which to extract reconstructed
                features.

        Returns:
            Dictionary of reconstructed features for all pulses in `pulsemap`,
                in pure-python format.
        """
        output: Dict[str, List[Any]] = {
            "truth_flag": [],
        }

        # Add features from IceCubeUpgrade
        output_icecube_upgrade = super().__call__(frame)
        output.update(output_icecube_upgrade)

        # Get OM data
        if self._pulsemap in frame:
            om_keys, data = get_om_keys_and_pulseseries(
                frame,
                self._pulsemap,
                self._calibration,
            )
        else:
            self.warning_once(f"Pulsemap {self._pulsemap} not found in frame.")
            return output

        for om_key in om_keys:
            # Loop over pulses for each OM
            pulses = data[om_key]
            for truth_flag in pulses:
                output["truth_flag"].append(truth_flag)

        return output



## ToDo: check if the hardcoded angles are correct. and if this pmt posn logic is correct.