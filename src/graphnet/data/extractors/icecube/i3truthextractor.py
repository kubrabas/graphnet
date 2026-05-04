"""I3Extractor class(es) for extracting truth-level information."""

import numpy as np
import matplotlib.path as mpath
from scipy.spatial import ConvexHull, Delaunay
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .i3extractor import I3Extractor
from .utilities.frames import (
    frame_is_montecarlo,
    frame_is_noise,
)
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (  # noqa: F401
        dataclasses,
        icetray,
        phys_services,
        dataio,
        LeptonInjector,
    )  # pyright: reportMissingImports=false



class I3TruthExtractor(I3Extractor):
    """Class for extracting truth-level information."""

    def __init__(
        self,
        name: str = "truth",
        borders: Optional[List[np.ndarray]] = None,
        mctree: Optional[str] = "I3MCTree",
        extend_boundary: Optional[float] = 0.0,
        exclude: list = [None],
    ):
        """Construct I3TruthExtractor.

        Args:
            name: Name of the `I3Extractor` instance.
            borders: Array of boundaries of the detector volume as ((x,y),z)-
                coordinates, for identifying, e.g., particles starting and
                stopping within the detector. Defaults to hard-coded boundary
                coordinates.
            mctree: Str of which MCTree to use for truth values.
            extend_boundary: Distance to extend the convex hull of the detector
                for defining starting events.
            exclude: List of keys to exclude from the extracted data.
        """
        # Base class constructor
        super().__init__(name, exclude=exclude)

        if borders is None:
            border_xy = np.array(
                [
                    (-256.1400146484375, -521.0800170898438),
                    (-132.8000030517578, -501.45001220703125),
                    (-9.13000011444092, -481.739990234375),
                    (114.38999938964844, -461.989990234375),
                    (237.77999877929688, -442.4200134277344),
                    (361.0, -422.8299865722656),
                    (405.8299865722656, -306.3800048828125),
                    (443.6000061035156, -194.16000366210938),
                    (500.42999267578125, -58.45000076293945),
                    (544.0700073242188, 55.88999938964844),
                    (576.3699951171875, 170.9199981689453),
                    (505.2699890136719, 257.8800048828125),
                    (429.760009765625, 351.0199890136719),
                    (338.44000244140625, 463.7200012207031),
                    (224.5800018310547, 432.3500061035156),
                    (101.04000091552734, 412.7900085449219),
                    (22.11000061035156, 509.5),
                    (-101.05999755859375, 490.2200012207031),
                    (-224.08999633789062, 470.8599853515625),
                    (-347.8800048828125, 451.5199890136719),
                    (-392.3800048828125, 334.239990234375),
                    (-437.0400085449219, 217.8000030517578),
                    (-481.6000061035156, 101.38999938964844),
                    (-526.6300048828125, -15.60000038146973),
                    (-570.9000244140625, -125.13999938964844),
                    (-492.42999267578125, -230.16000366210938),
                    (-413.4599914550781, -327.2699890136719),
                    (-334.79998779296875, -424.5),
                ]
            )
            border_z = np.array([-512.82, 524.56])
            self._borders = [border_xy, border_z]
        else:
            self._borders = borders

        self._extend_boundary = extend_boundary
        self._mctree = mctree

    def set_gcd(self, i3_file: str, gcd_file: Optional[str] = None) -> None:
        """Extract GFrame and CFrame from i3/gcd-file pair.

           Information from these frames will be set as member variables of
           `I3Extractor.`

        Args:
            i3_file: Path to i3 file that is being converted.
            gcd_file: Path to GCD file. Defaults to None. If no GCD file is
                      given, the method will attempt to find C and G frames in
                      the i3 file instead. If either one of those are not
                      present, `RuntimeErrors` will be raised.
        """
        super().set_gcd(i3_file=i3_file, gcd_file=gcd_file)

        # Modifications specific to I3TruthExtractor
        # These modifications are needed to identify starting events
        coordinates = []
        for _, g in self._gcd_dict.items():
            if g.position.z > 1200:
                continue  # We want to exclude icetop
            coordinates.append([g.position.x, g.position.y, g.position.z])
        coordinates = np.array(coordinates)

        if self._extend_boundary != 0.0:
            center = np.mean(coordinates, axis=0)
            d = coordinates - center
            norms = np.linalg.norm(d, axis=1, keepdims=True)
            dn = d / norms
            coordinates = coordinates + dn * self._extend_boundary

        hull = ConvexHull(coordinates)

        self.hull = hull
        self.delaunay = Delaunay(coordinates[self.hull.vertices])

    def __call__(
        self, frame: "icetray.I3Frame", padding_value: Any = -1
    ) -> Dict[str, Any]:
        """Extract truth-level information."""
        is_mc = frame_is_montecarlo(frame, self._mctree)
        is_noise = frame_is_noise(frame, self._mctree)
        sim_type = self._find_data_type(is_mc, self._i3_file, frame)

        output = {
            "energy": padding_value,
            "position_x": padding_value,
            "position_y": padding_value,
            "position_z": padding_value,
            "azimuth": padding_value,
            "zenith": padding_value,
            "pid": padding_value,
            "event_time": frame["I3EventHeader"].start_time.utc_daq_time,
            "sim_type": sim_type,
            "interaction_type": padding_value,
            "elasticity": padding_value,
            "RunID": frame["I3EventHeader"].run_id,
            "SubrunID": frame["I3EventHeader"].sub_run_id,
            "EventID": frame["I3EventHeader"].event_id,
            "SubEventID": frame["I3EventHeader"].sub_event_id,
            "dbang_decay_length": padding_value,
            "track_length": padding_value,
            "stopped_muon": padding_value,
            "energy_track": padding_value,
            "energy_cascade": padding_value,
            "inelasticity": padding_value,
            "DeepCoreFilter_13": padding_value,
            "CascadeFilter_13": padding_value,
            "MuonFilter_13": padding_value,
            "OnlineL2Filter_17": padding_value,
            "L3_oscNext_bool": padding_value,
            "L4_oscNext_bool": padding_value,
            "L5_oscNext_bool": padding_value,
            "L6_oscNext_bool": padding_value,
            "L7_oscNext_bool": padding_value,
            "is_starting": padding_value,
        }

        # Only InIceSplit P frames contain ML appropriate
        # for example I3RecoPulseSeriesMap,  etc.
        # At low levels i3 files contain several other P frame splits
        # (e.g NullSplit). We remove those here.
        if frame["I3EventHeader"].sub_event_stream not in [
            "InIceSplit",
            "Final",
        ]:
            return output

        if "FilterMask" in frame:
            if "DeepCoreFilter_13" in frame["FilterMask"]:
                output["DeepCoreFilter_13"] = int(
                    bool(frame["FilterMask"]["DeepCoreFilter_13"])
                )
            if "CascadeFilter_13" in frame["FilterMask"]:
                output["CascadeFilter_13"] = int(
                    bool(frame["FilterMask"]["CascadeFilter_13"])
                )
            if "MuonFilter_13" in frame["FilterMask"]:
                output["MuonFilter_13"] = int(
                    bool(frame["FilterMask"]["MuonFilter_13"])
                )
            if "OnlineL2Filter_17" in frame["FilterMask"]:
                output["OnlineL2Filter_17"] = int(
                    bool(frame["FilterMask"]["OnlineL2Filter_17"])
                )

        elif "DeepCoreFilter_13" in frame:
            output["DeepCoreFilter_13"] = int(bool(frame["DeepCoreFilter_13"]))

        if "L3_oscNext_bool" in frame:
            output["L3_oscNext_bool"] = int(bool(frame["L3_oscNext_bool"]))

        if "L4_oscNext_bool" in frame:
            output["L4_oscNext_bool"] = int(bool(frame["L4_oscNext_bool"]))

        if "L5_oscNext_bool" in frame:
            output["L5_oscNext_bool"] = int(bool(frame["L5_oscNext_bool"]))

        if "L6_oscNext_bool" in frame:
            output["L6_oscNext_bool"] = int(bool(frame["L6_oscNext_bool"]))

        if "L7_oscNext_bool" in frame:
            output["L7_oscNext_bool"] = int(bool(frame["L7_oscNext_bool"]))

        if is_mc and (not is_noise):
            (
                MCInIcePrimary,
                interaction_type,
                elasticity,
            ) = self._get_primary_particle_interaction_type_and_elasticity(
                frame, sim_type
            )

            try:
                (
                    energy_track,
                    energy_cascade,
                    inelasticity,
                ) = self._get_primary_track_energy_and_inelasticity(frame)
            except (
                RuntimeError
            ):  # track energy fails on northeren tracks with ""Hadrons"
                # has no mass implemented. Cannot get total energy."
                energy_track, energy_cascade, inelasticity = (
                    padding_value,
                    padding_value,
                    padding_value,
                )

            output.update(
                {
                    "energy": MCInIcePrimary.energy,
                    "position_x": MCInIcePrimary.pos.x,
                    "position_y": MCInIcePrimary.pos.y,
                    "position_z": MCInIcePrimary.pos.z,
                    "azimuth": MCInIcePrimary.dir.azimuth,
                    "zenith": MCInIcePrimary.dir.zenith,
                    "pid": MCInIcePrimary.pdg_encoding,
                    "interaction_type": interaction_type,
                    "elasticity": elasticity,
                    "dbang_decay_length": self._extract_dbang_decay_length(
                        frame, padding_value
                    ),
                    "energy_track": energy_track,
                    "energy_cascade": energy_cascade,
                    "inelasticity": inelasticity,
                }
            )
            if abs(output["pid"]) == 13:
                output.update(
                    {
                        "track_length": MCInIcePrimary.length,
                    }
                )
                muon_final = self._muon_stopped(output, self._borders)
                output.update(
                    {
                        "position_x": muon_final["x"],
                        # position_xyz has no meaning for muons.
                        # These will now be updated to muon final position,
                        # given track length/azimuth/zenith
                        "position_y": muon_final["y"],
                        "position_z": muon_final["z"],
                        "stopped_muon": muon_final["stopped"],
                    }
                )

            is_starting = self._contained_vertex(output)
            output.update(
                {
                    "is_starting": is_starting,
                }
            )

        return output

    def _extract_dbang_decay_length(
        self, frame: "icetray.I3Frame", padding_value: float = -1
    ) -> float:
        mctree = frame[self._mctree]
        try:
            p_true = mctree.primaries[0]
            p_daughters = mctree.get_daughters(p_true)
            if len(p_daughters) == 2:
                for p_daughter in p_daughters:
                    if p_daughter.type == dataclasses.I3Particle.Hadrons:
                        casc_0_true = p_daughter
                    else:
                        hnl_true = p_daughter
                hnl_daughters = mctree.get_daughters(hnl_true)
            else:
                decay_length = padding_value
                hnl_daughters = []

            if len(hnl_daughters) > 0:
                for count_hnl_daughters, hnl_daughter in enumerate(
                    hnl_daughters
                ):
                    if not count_hnl_daughters:
                        casc_1_true = hnl_daughter
                    else:
                        assert casc_1_true.pos == hnl_daughter.pos
                        casc_1_true.energy = (
                            casc_1_true.energy + hnl_daughter.energy
                        )
                decay_length = (
                    phys_services.I3Calculator.distance(
                        casc_0_true, casc_1_true
                    )
                    / icetray.I3Units.m
                )

            else:
                decay_length = padding_value
            return decay_length
        except:  # noqa: E722
            return padding_value

    def _muon_stopped(
        self,
        truth: Dict[str, Any],
        borders: List[np.ndarray],
        shrink_horizontally: float = 100.0,
        shrink_vertically: float = 100.0,
    ) -> Dict[str, Any]:
        """Calculate whether a simulated muon within the detector volume.

        IMPORTANT: The final position of the muon is saved in truth extractor/
        databases as position_x, position_y and position_z. This is analogouos
        to the neutrinos whose interaction vertex is saved under the same name.

        Args:
            truth: Dictionary of already extracted truth-level information.
            borders: The first entry are the (x,y) coordinates, the second
                entry is the z-axis min/max depths. See I3TruthExtractor
                constructor for hard-code example.
            shrink_horizontally: Shrink (x,y)-plane further with exclusion
                zone. Defaults to 100 meters. shrink_vertically: Further shrink
                detector depth with exclusion height. Defaults to 100 meters.

        Returns:
            Dictionary containing the (x,y,z)-coordinates of final the muon
                position as well as a boolean indicating whether the muon
                stopped within the chosen fiducial volume.
        """
        # @TODO: Remove hard-coded border coords and replace with GCD file
        # contents using string no's
        border = mpath.Path(borders[0])

        start_pos = np.array(
            [truth["position_x"], truth["position_y"], truth["position_z"]]
        )

        travel_vec = -1 * np.array(
            [
                truth["track_length"]
                * np.cos(truth["azimuth"])
                * np.sin(truth["zenith"]),
                truth["track_length"]
                * np.sin(truth["azimuth"])
                * np.sin(truth["zenith"]),
                truth["track_length"] * np.cos(truth["zenith"]),
            ]
        )

        end_pos = start_pos + travel_vec

        stopped_xy = border.contains_point(
            (end_pos[0], end_pos[1]), radius=-shrink_horizontally
        )
        stopped_z = (end_pos[2] > borders[1][0] + shrink_vertically) * (
            end_pos[2] < borders[1][1] - shrink_vertically
        )

        return {
            "x": end_pos[0],
            "y": end_pos[1],
            "z": end_pos[2],
            "stopped": (stopped_xy * stopped_z),
        }

    def _get_primary_particle_interaction_type_and_elasticity(
        self,
        frame: "icetray.I3Frame",
        sim_type: str,
        padding_value: float = -1.0,
    ) -> Tuple[Any, int, float]:
        """Return primary particle, interaction type, and elasticity.

        A case handler that does two things:
            1) Catches issues related to determining the primary MC particle.
            2) Error handles cases where interaction type and elasticity
                doesn't exist

        Args:
            frame: Physics frame containing MC record.
            sim_type: Simulation type.
            padding_value: The value used for padding.

        Returns
            A tuple containing the MCInIcePrimary, if it exists; the primary
                particle, encoded as 1 (charged current), 2 (neutral current),
                or 0 (neither); and the elasticity in the range ]0,1[.
        """
        if sim_type != "noise":
            try:
                MCInIcePrimary = frame["MCInIcePrimary"]
            except KeyError:
                MCInIcePrimary = frame[self._mctree][0]
            if (
                MCInIcePrimary.energy != MCInIcePrimary.energy
            ):  # This is a nan check. Only happens for some muons
                # where second item in MCTree is primary. Weird!
                MCInIcePrimary = frame[self._mctree][1]
                # For some strange reason the second entry is identical in
                # all variables and has no nans (always muon)
        else:
            MCInIcePrimary = None

        if sim_type == "LeptonInjector":
            event_properties = frame["EventProperties"]
            final_state_1 = event_properties.finalType1
            if final_state_1 in [
                dataclasses.I3Particle.NuE,
                dataclasses.I3Particle.NuMu,
                dataclasses.I3Particle.NuTau,
                dataclasses.I3Particle.NuEBar,
                dataclasses.I3Particle.NuMuBar,
                dataclasses.I3Particle.NuTauBar,
            ]:
                interaction_type = 2  # NC
            else:
                interaction_type = 1  # CC

            elasticity = 1 - event_properties.finalStateY

        else:
            try:
                interaction_type = frame["I3MCWeightDict"]["InteractionType"]
            except KeyError:
                interaction_type = int(padding_value)

            try:
                elasticity = 1 - frame["I3MCWeightDict"]["BjorkenY"]
            except KeyError:
                elasticity = padding_value

        return MCInIcePrimary, interaction_type, elasticity

    def _get_primary_track_energy_and_inelasticity(
        self,
        frame: "icetray.I3Frame",
    ) -> Tuple[float, float, float]:
        """Get the total energy of tracks from primary, and inelasticity.

        Args:
            frame: Physics frame containing MC record.

        Returns:
            Tuple containing the energy of tracks from primary, and the
                corresponding inelasticity.
        """
        mc_tree = frame[self._mctree]
        primary = mc_tree.primaries[0]
        daughters = mc_tree.get_daughters(primary)
        tracks = []
        for daughter in daughters:
            if (
                str(daughter.shape) == "StartingTrack"
                or str(daughter.shape) == "Dark"
            ):
                tracks.append(daughter)

        energy_total = primary.total_energy
        energy_track = sum(track.total_energy for track in tracks)
        energy_cascade = energy_total - energy_track
        inelasticity = 1.0 - energy_track / energy_total

        return energy_track, energy_cascade, inelasticity

    # Utility methods
    def _find_data_type(
        self, mc: bool, input_file: str, frame: "icetray.I3Frame"
    ) -> str:
        """Determine the data type.

        Args:
            mc: Whether `input_file` is Monte Carlo simulation.
            input_file: Path to I3-file.
            frame: Physics frame containing MC record

        Returns:
            The simulation/data type.
        """
        # @TODO: Rewrite to automatically infer `mc` from `input_file`?
        if not mc:
            sim_type = "data"
        elif "muon" in input_file:
            sim_type = "muongun"
        elif "corsika" in input_file:
            sim_type = "corsika"
        elif "genie" in input_file or "nu" in input_file.lower():
            sim_type = "genie"
        elif "noise" in input_file:
            sim_type = "noise"
        elif frame.Has("EventProprties") or frame.Has(
            "LeptonInjectorProperties"
        ):
            sim_type = "LeptonInjector"
        elif frame.Has("I3MCWeightDict"):
            sim_type = "NuGen"
        else:
            raise NotImplementedError("Could not determine data type.")
        return sim_type

    def _contained_vertex(self, truth: Dict[str, Any]) -> bool:
        """Determine if an event is starting based on vertex position.

        Args:
            truth: Dictionary of already extracted truth-level information.

        Returns:
            True/False if vertex is inside detector.
        """
        vertex = np.array(
            [truth["position_x"], truth["position_y"], truth["position_z"]]
        )
        return self.delaunay.find_simplex(vertex) >= 0



#### check if the truth extractor correct as you want:
class I3TruthExtractorPONE(I3Extractor):
    """Truth extractor for PONE/LeptonInjector simulations.

    Extracts per-event kinematics from EventProperties and computes five
    mutually exclusive muon track containment flags for NuMu/NuMuBar CC events.

    All containment flags describe the muon's path relative to the detector
    convex hull (built from the GCD file passed to ``set_gcd``).

    ---

    Coordinate sources
    ------------------
    ``ep.x / ep.y / ep.z``  (from ``EventProperties``)
        The neutrino interaction vertex — equivalently, the point where the
        muon is born.  Used as the muon's **birth position**.

    ``mmc.xi / mmc.yi / mmc.zi``  (from ``MMCTrackList``)
        The point where the muon crosses into the PROPOSAL simulation
        cylinder.  This cylinder is defined around the *full* detector
        geometry used in the simulation, which may be larger than the
        sub-geometry GCD hull used here.  Used only for the
        ``through_going`` segment sampling (not for containment checks).

    ``mmc.xf / mmc.yf / mmc.zf``  (from ``MMCTrackList``)
        Either the muon's **actual stopping point** (if it runs out of
        energy inside the PROPOSAL cylinder) or the point where it
        **exits** the PROPOSAL cylinder (if it is still energetic).
        In both cases, checking this point against the detector hull
        correctly answers "did the muon stop inside the detector?":
        a stopped muon gives its true final position; a cylinder-exiting
        muon gives a point that is outside the (smaller) detector hull.
        Used as the muon's **end position**.

    ---

    Containment flags  (NuMu / NuMuBar CC only)
    --------------------------------------------
    All flags are binary (0 or 1) and mutually exclusive.
    They are set to the padding value (-1) for non-NuMu events or
    when ``MMCTrackList`` is absent from the frame.

    ``fully_contained``
        birth inside detector  AND  end inside detector.
        The muon is created and stops entirely within the hull.

    ``starting_track``
        birth inside detector  AND  end outside detector.
        The muon is created inside the hull but exits before stopping.

    ``stopping_track``
        birth outside detector  AND  end inside detector.
        The muon enters the hull from outside and stops inside.

    ``through_going``
        birth outside detector  AND  end outside detector
        AND the MMC segment (xi→xf) intersects the hull.
        The muon passes through the hull without stopping inside.

    ``missed_track``
        birth outside detector  AND  end outside detector
        AND the MMC segment (xi→xf) does not intersect the hull.
        The muon never enters the detector volume.
    """

    def __init__(
        self,
        name: str = "truth",
        extend_boundary: Optional[float] = 0.0,
        exclude: list = [None],
    ):
        """Construct I3TruthExtractorPONE.

        Args:
            name: Name of the `I3Extractor` instance.
            extend_boundary: Distance in metres to expand the convex hull of
                the detector when testing vertex containment. Defaults to 0.
            exclude: List of keys to exclude from the extracted data.
        """
        super().__init__(name, exclude=exclude)
        self._extend_boundary = extend_boundary


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

        coordinates = np.array([
            [g.position.x, g.position.y, g.position.z]
            for g in self._gcd_dict.values()
        ])


        if self._extend_boundary != 0.0:
            center = np.mean(coordinates, axis=0)
            d = coordinates - center
            norms = np.linalg.norm(d, axis=1, keepdims=True)
            coordinates = coordinates + (d / norms) * self._extend_boundary

        hull = ConvexHull(coordinates)
        self.hull = hull
        self.delaunay = Delaunay(coordinates[hull.vertices])


    
    def __call__(
        self, frame: "icetray.I3Frame", padding_value: Any = -1
    ) -> Dict[str, Any]:
        """Extract truth-level information for a single DAQ frame."""
        output = {
            # I3EventHeader
            "RunID":      padding_value,
            "SubrunID":   padding_value,
            "EventID":    padding_value,
            "SubEventID": padding_value,
            # EventProperties
            "position_x":       padding_value,
            "position_y":       padding_value,
            "position_z":       padding_value,
            "pid":              padding_value,
            "interaction_type": padding_value,
            "totalEnergy":      padding_value,
            "zenith":           padding_value,
            "azimuth":          padding_value,
            "finalStateX":      padding_value,
            "finalStateY":      padding_value,
            "finalType1":       padding_value,
            "finalType2":       padding_value,
            "initialType":      padding_value,
            "totalColumnDepth": padding_value,
            "impactParameter":  padding_value,
            # Muon containment flags (NuMu/NuMuBar CC only — see class docstring)
            "fully_contained": padding_value,
            "starting_track":  padding_value,
            "stopping_track":  padding_value,
            "through_going":   padding_value,
            "missed_track":    padding_value,
        }

        if len(frame) == 0:
            print("[I3TruthExtractorPONE] Empty frame, skipping.")
            return output

        required = ["I3EventHeader", "EventProperties"]
        missing = [k for k in required if k not in frame]
        if missing:
            print(
                f"[I3LeptonInjectorExtractorPONE] Missing frame keys {missing}, skipping."
            )
            return output

        # I3EventHeader
        header = frame["I3EventHeader"]
        output.update(
            {
                "RunID": header.run_id,
                "SubrunID": header.sub_run_id,
                "EventID": header.event_id,
                "SubEventID": header.sub_event_id,
            }
        )

        # EventProperties
        ep = frame["EventProperties"]
        output.update(
            {
                "position_x": ep.x,
                "position_y": ep.y,
                "position_z": ep.z,
                "pid": int(ep.initialType),
                "totalEnergy": ep.totalEnergy,
                "zenith": ep.zenith,
                "azimuth": ep.azimuth,
                "finalStateX": ep.finalStateX,
                "finalStateY": ep.finalStateY,
                "finalType1": ep.finalType1,
                "finalType2": ep.finalType2,
                "initialType": ep.initialType,
                "interaction_type": self._get_interaction_type(ep),
            }
        )
        try:
            output["totalColumnDepth"] = ep.totalColumnDepth
            output["impactParameter"] = ep.impactParameter
        except AttributeError:
            pass  

        # Containment flags — only for NuMu/NuMuBar CC events (muon tracks)
        if abs(int(ep.initialType)) == 14:
            mmc_list = frame["MMCTrackList"] if "MMCTrackList" in frame else []

            if len(mmc_list) > 0:
                mmc = mmc_list[0]
                birth  = np.array([ep.x, ep.y, ep.z])
                mmc_entry = np.array([mmc.xi, mmc.yi, mmc.zi])
                end    = np.array([mmc.xf, mmc.yf, mmc.zf])

                birth_inside = self._inside_detector(birth)
                end_inside   = self._inside_detector(end)

                through_going, missed_track = 0, 0
                if not birth_inside and not end_inside:
                    passes = any(
                        self._inside_detector(mmc_entry + t * (end - mmc_entry))
                        for t in np.linspace(0, 1, 100)
                    )
                    through_going = int(passes)
                    missed_track  = int(not passes)

                output.update(
                    {
                        "fully_contained": int(birth_inside and end_inside),
                        "starting_track":  int(birth_inside and not end_inside),
                        "stopping_track":  int(not birth_inside and end_inside),
                        "through_going":   through_going,
                        "missed_track":    missed_track,
                    }
                )

        return output

    def _get_interaction_type(self, ep: Any) -> int:
        """Return 1 (CC) or 2 (NC) based on EventProperties finalType1."""
        neutrinos = [
            dataclasses.I3Particle.NuE,
            dataclasses.I3Particle.NuMu,
            dataclasses.I3Particle.NuTau,
            dataclasses.I3Particle.NuEBar,
            dataclasses.I3Particle.NuMuBar,
            dataclasses.I3Particle.NuTauBar,
        ]
        return 2 if ep.finalType1 in neutrinos else 1

    def _inside_detector(self, point: np.ndarray) -> bool:
        """Return True if point is inside the detector convex hull."""
        return bool(self.delaunay.find_simplex(point) >= 0)
