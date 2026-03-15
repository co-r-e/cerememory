//! Media format normalization helpers shared across LLM adapters.

use crate::error::CerememoryError;

/// Normalize a user-supplied image format string to a canonical MIME type.
///
/// Accepts both short names ("png", "jpg") and full MIME types ("image/png").
/// Returns a static MIME string suitable for HTTP headers and data URIs.
pub fn normalize_image_mime_type(format: &str) -> Result<&'static str, CerememoryError> {
    match format.trim().to_ascii_lowercase().as_str() {
        "png" | "image/png" => Ok("image/png"),
        "jpg" | "jpeg" | "image/jpg" | "image/jpeg" => Ok("image/jpeg"),
        "gif" | "image/gif" => Ok("image/gif"),
        "webp" | "image/webp" => Ok("image/webp"),
        other => Err(CerememoryError::Validation(format!(
            "Unsupported image format: {other}"
        ))),
    }
}

/// Normalize a user-supplied audio format string to a canonical
/// `(file_extension, mime_type)` pair.
///
/// Accepts both short names ("wav", "mp3") and full MIME types ("audio/wav").
pub fn normalize_audio_upload_format(
    format: &str,
) -> Result<(&'static str, &'static str), CerememoryError> {
    match format.trim().to_ascii_lowercase().as_str() {
        "wav" | "wave" | "audio/wav" | "audio/x-wav" => Ok(("wav", "audio/wav")),
        "mp3" | "audio/mp3" | "audio/mpeg" | "mpeg" | "mpga" => Ok(("mp3", "audio/mpeg")),
        "m4a" | "audio/mp4" | "audio/x-m4a" => Ok(("m4a", "audio/mp4")),
        "mp4" => Ok(("mp4", "audio/mp4")),
        "flac" | "audio/flac" | "audio/x-flac" => Ok(("flac", "audio/flac")),
        "ogg" | "oga" | "audio/ogg" => Ok(("ogg", "audio/ogg")),
        "webm" | "audio/webm" => Ok(("webm", "audio/webm")),
        other => Err(CerememoryError::Validation(format!(
            "Unsupported audio format: {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_image_accepts_short_names() {
        assert_eq!(normalize_image_mime_type("png").unwrap(), "image/png");
        assert_eq!(normalize_image_mime_type("jpg").unwrap(), "image/jpeg");
        assert_eq!(normalize_image_mime_type("jpeg").unwrap(), "image/jpeg");
        assert_eq!(normalize_image_mime_type("gif").unwrap(), "image/gif");
        assert_eq!(normalize_image_mime_type("webp").unwrap(), "image/webp");
    }

    #[test]
    fn normalize_image_accepts_mime_types() {
        assert_eq!(normalize_image_mime_type("image/png").unwrap(), "image/png");
        assert_eq!(
            normalize_image_mime_type("image/jpeg").unwrap(),
            "image/jpeg"
        );
    }

    #[test]
    fn normalize_image_rejects_unknown() {
        assert!(normalize_image_mime_type("bmp").is_err());
    }

    #[test]
    fn normalize_audio_accepts_short_names() {
        assert_eq!(
            normalize_audio_upload_format("wav").unwrap(),
            ("wav", "audio/wav")
        );
        assert_eq!(
            normalize_audio_upload_format("mp3").unwrap(),
            ("mp3", "audio/mpeg")
        );
        assert_eq!(
            normalize_audio_upload_format("flac").unwrap(),
            ("flac", "audio/flac")
        );
    }

    #[test]
    fn normalize_audio_accepts_mime_types() {
        assert_eq!(
            normalize_audio_upload_format("audio/wav").unwrap(),
            ("wav", "audio/wav")
        );
        assert_eq!(
            normalize_audio_upload_format("audio/mpeg").unwrap(),
            ("mp3", "audio/mpeg")
        );
    }

    #[test]
    fn normalize_audio_rejects_unknown() {
        assert!(normalize_audio_upload_format("application/octet-stream").is_err());
    }
}
