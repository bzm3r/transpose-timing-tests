use std::fs;
use std::path::Path;

pub fn is_relatively_fresh(base: &Path, other: &Path) -> bool {
    if base.exists() && other.exists() {
        let base_meta = fs::metadata(base).unwrap();
        let other_meta = fs::metadata(other).unwrap();

        if base_meta.modified().unwrap() < other_meta.modified().unwrap() {
            return true;
        }
    }
    false
}
