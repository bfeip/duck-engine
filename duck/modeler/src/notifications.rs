//! Transient user-facing notices (toasts), shared between tools and the UI.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long a notice stays on screen.
const NOTICE_TTL: Duration = Duration::from_secs(6);
/// Oldest notices are dropped beyond this count.
const MAX_NOTICES: usize = 5;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Info,
    Error,
}

#[derive(Clone)]
pub struct Notice {
    pub text: String,
    pub severity: Severity,
    pub created: Instant,
}

/// Cloneable handle to the shared notice queue. Tools push notices from
/// anywhere; the UI drains expired ones each frame via [`Notifications::live`].
#[derive(Clone, Default)]
pub struct Notifications {
    queue: Arc<Mutex<VecDeque<Notice>>>,
}

impl Notifications {
    pub fn error(&self, text: impl Into<String>) {
        self.push(text.into(), Severity::Error);
    }

    pub fn info(&self, text: impl Into<String>) {
        self.push(text.into(), Severity::Info);
    }

    fn push(&self, text: String, severity: Severity) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(Notice { text, severity, created: Instant::now() });
        while queue.len() > MAX_NOTICES {
            queue.pop_front();
        }
    }

    /// Prune expired notices and return the live ones, oldest first.
    pub fn live(&self) -> Vec<Notice> {
        let mut queue = self.queue.lock().unwrap();
        let now = Instant::now();
        queue.retain(|n| now.duration_since(n.created) < NOTICE_TTL);
        queue.iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notices_appear_and_cap() {
        let notifications = Notifications::default();
        for i in 0..(MAX_NOTICES + 2) {
            notifications.error(format!("notice {i}"));
        }
        let live = notifications.live();
        assert_eq!(live.len(), MAX_NOTICES, "queue must cap at MAX_NOTICES");
        assert_eq!(live[0].text, "notice 2", "oldest notices are dropped first");
    }
}
