//! Browser -> Rust -> Python control-message plumbing.
//!
//! Dev-guide §6.3: control messages travel as JSON text frames over the
//! WebSocket. Keeping them JSON (not msgpack) lets the user inspect them
//! live in devtools and lets the Python side consume them with the
//! stdlib `json` module -- no msgpack dep on the control path.
//!
//! The enum is externally tagged (`{"type": "set_heat_flux", "value": 30000}`)
//! so the on-the-wire shape matches the TypeScript `ControlMessage`
//! literal union from the dev guide verbatim.

use serde::{Deserialize, Serialize};

/// Every control message the browser may send. Variant names use
/// `snake_case` on the wire to match the TypeScript discriminant values
/// (`set_heat_flux`, `set_material`, etc.) documented in the dev guide.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlMessage {
    SetHeatFlux { value: f32 },
    SetMaterial { value: String },
    SetCarrotSize { diameter_mm: f32, length_mm: f32 },
    /// Swap the solute being tracked. `value` is one of
    /// `"beta_carotene"`, `"vitamin_c"`, or `"both"`; the Python side
    /// applies the matching parameter preset and rebuilds the
    /// Simulation. Added in schema v2 after user feedback that the
    /// dashboard hard-coded β-carotene and couldn't test vitamin C.
    SetNutrient { value: String },
    /// Phase 6.6: stage a full `ScenarioConfig` from the Configuration
    /// page. `config` is an opaque JSON blob (Pydantic validation
    /// happens on the Python side); on validation failure the Python
    /// producer surfaces the error in `Snapshot.last_error` and keeps
    /// the current cfg. On success, a full Simulation rebuild fires.
    SetConfig { config: serde_json::Value },
    /// Phase 6.6: begin a new timed run. Resets `run_id`, clears the
    /// scalar history + `is_complete` flag, sets `total_time_s`, and
    /// resumes stepping. Usually follows a `SetConfig` from the
    /// Configuration page's "Apply & Start Run" button.
    StartRun { duration_s: f32 },
    /// Phase 6.6: emit the three run artefacts (HDF5 / CSV / JSON)
    /// for the current run-in-progress without resetting or pausing.
    /// Useful for "save what we have so far" during a long run.
    ExportSnapshot,
    Pause,
    Resume,
    Reset,
    /// Stop the run mid-flight, write the partial-history artefacts,
    /// and flip `is_complete` so the Results page becomes available.
    /// Distinct from `ExportSnapshot` (which keeps stepping) and
    /// `Reset` (which discards the run). Used when the user has seen
    /// enough and wants to inspect results without waiting for
    /// `total_time_s`.
    Finalize,
    RequestFullSnapshot,
}

impl ControlMessage {
    /// Parse a JSON text frame from the browser.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Encode for forwarding to Python. We keep JSON here too so Python
    /// can decode with `json.loads` and there is exactly one text-based
    /// wire shape for control traffic.
    pub fn to_json_line(&self) -> String {
        // Push a newline so a line-delimited reader on the Python side
        // (which is what `run_dashboard.py` will use in M3) can cheaply
        // split frames without a length prefix.
        let mut s = serde_json::to_string(self).expect("control message serialises");
        s.push('\n');
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_set_heat_flux() {
        let cmd = ControlMessage::SetHeatFlux { value: 30000.0 };
        let json = serde_json::to_string(&cmd).unwrap();
        assert_eq!(json, r#"{"type":"set_heat_flux","value":30000.0}"#);
        let back: ControlMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cmd);
    }

    #[test]
    fn round_trip_set_material() {
        let cmd = ControlMessage::SetMaterial { value: "copper".into() };
        let json = serde_json::to_string(&cmd).unwrap();
        assert_eq!(json, r#"{"type":"set_material","value":"copper"}"#);
        assert_eq!(ControlMessage::from_json(&json).unwrap(), cmd);
    }

    #[test]
    fn round_trip_set_carrot_size() {
        let cmd = ControlMessage::SetCarrotSize {
            diameter_mm: 25.0,
            length_mm: 50.0,
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains(r#""type":"set_carrot_size""#));
        assert_eq!(ControlMessage::from_json(&json).unwrap(), cmd);
    }

    #[test]
    fn round_trip_set_config_carries_json_blob() {
        let cmd = ControlMessage::SetConfig {
            config: serde_json::json!({
                "pot": {"material": "copper", "diameter_m": 0.18},
                "total_time_s": 120.0,
            }),
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains(r#""type":"set_config""#));
        let back: ControlMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cmd);
        // Inner blob survives roundtrip.
        if let ControlMessage::SetConfig { config } = back {
            assert_eq!(config["pot"]["material"], "copper");
            assert_eq!(config["total_time_s"], 120.0);
        } else {
            panic!("expected SetConfig");
        }
    }

    #[test]
    fn round_trip_start_run() {
        let cmd = ControlMessage::StartRun { duration_s: 600.0 };
        let json = serde_json::to_string(&cmd).unwrap();
        assert_eq!(json, r#"{"type":"start_run","duration_s":600.0}"#);
        assert_eq!(ControlMessage::from_json(&json).unwrap(), cmd);
    }

    #[test]
    fn round_trip_export_snapshot() {
        let cmd = ControlMessage::ExportSnapshot;
        let json = serde_json::to_string(&cmd).unwrap();
        assert_eq!(json, r#"{"type":"export_snapshot"}"#);
        assert_eq!(ControlMessage::from_json(&json).unwrap(), cmd);
    }

    #[test]
    fn parse_pause_resume_reset() {
        for (raw, expected) in [
            (r#"{"type":"pause"}"#, ControlMessage::Pause),
            (r#"{"type":"resume"}"#, ControlMessage::Resume),
            (r#"{"type":"reset"}"#, ControlMessage::Reset),
            (r#"{"type":"finalize"}"#, ControlMessage::Finalize),
            (r#"{"type":"request_full_snapshot"}"#, ControlMessage::RequestFullSnapshot),
        ] {
            assert_eq!(ControlMessage::from_json(raw).unwrap(), expected);
        }
    }

    #[test]
    fn round_trip_finalize() {
        let cmd = ControlMessage::Finalize;
        let json = serde_json::to_string(&cmd).unwrap();
        assert_eq!(json, r#"{"type":"finalize"}"#);
        assert_eq!(ControlMessage::from_json(&json).unwrap(), cmd);
    }

    #[test]
    fn unknown_type_is_rejected() {
        let result = ControlMessage::from_json(r#"{"type":"nuke_simulation"}"#);
        assert!(result.is_err(), "unknown type should fail to parse");
    }

    #[test]
    fn to_json_line_is_newline_terminated() {
        let s = ControlMessage::Pause.to_json_line();
        assert!(s.ends_with('\n'));
        assert_eq!(s.trim_end(), r#"{"type":"pause"}"#);
    }
}
