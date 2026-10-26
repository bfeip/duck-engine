//! Values typed into a transform in place of the mouse's, Blender style.
//!
//! Digits and a point build the active field, `-` negates it, Backspace
//! deletes, and Tab moves to the next field. A field can also be set outright,
//! as a panel's value field does.

use duck_engine_viewer::common::Real;
use duck_engine_viewer::input::{Key, NamedKey};

/// A field's entry.
#[derive(Clone, Debug, PartialEq)]
enum Entry {
    /// Typed from the keyboard: the text as typed, sign included, and what it
    /// reads as.
    Typed { text: String, value: Real },
    /// Set outright.
    Set(Real),
}

impl Entry {
    fn value(&self) -> Real {
        match self {
            Entry::Typed { value, .. } | Entry::Set(value) => *value,
        }
    }
}

/// Up to three typed values, one of them taking keys.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TypedValues {
    entries: [Option<Entry>; 3],
    active: usize,
}

impl TypedValues {
    /// Whether `key` starts a value, as a digit, a point or a sign does.
    pub fn starts_value(key: &Key) -> bool {
        matches!(key, Key::Character(c) if c.is_ascii_digit() || matches!(c, '.' | '-'))
    }

    /// The value of `field`, or `None` when nothing is typed or set there.
    pub fn value(&self, field: usize) -> Option<Real> {
        self.entries.get(field)?.as_ref().map(Entry::value)
    }

    /// What was typed into `field`, while it was typed rather than set.
    pub fn text(&self, field: usize) -> Option<&str> {
        match self.entries.get(field)? {
            Some(Entry::Typed { text, .. }) => Some(text),
            _ => None,
        }
    }

    /// The field keys type into, among `fields`.
    pub fn active(&self, fields: usize) -> usize {
        self.active.min(fields.saturating_sub(1))
    }

    /// Sets `field` outright.
    pub fn set(&mut self, field: usize, value: Real) {
        if let Some(entry) = self.entries.get_mut(field) {
            *entry = Some(Entry::Set(value));
        }
    }

    /// Edits the active field, among `fields`, by `key`, reading typed text
    /// with `parse`; text that reads as no number yet counts as zero. `back`
    /// turns Tab into a step back. Returns whether `key` is one that types.
    pub fn type_key(
        &mut self,
        key: &Key,
        back: bool,
        fields: usize,
        parse: impl Fn(&str) -> Option<Real>,
    ) -> bool {
        if fields == 0 {
            return false;
        }
        let active = self.active(fields);
        let entry = &mut self.entries[active];
        match key {
            Key::Named(NamedKey::Tab) => {
                self.active = match back {
                    true => (active + fields - 1) % fields,
                    false => (active + 1) % fields,
                };
                return true;
            }
            Key::Named(NamedKey::Backspace) => {
                *entry = match entry.take() {
                    Some(Entry::Typed { mut text, .. }) => {
                        text.pop();
                        (!text.is_empty()).then(|| typed(text, &parse))
                    }
                    // A set value is one unit: it goes whole.
                    _ => None,
                };
            }
            Key::Character('-') => {
                *entry = Some(match entry.take() {
                    Some(Entry::Typed { text, .. }) => match text.strip_prefix('-') {
                        Some(positive) => typed(positive.to_owned(), &parse),
                        None => typed(format!("-{text}"), &parse),
                    },
                    Some(Entry::Set(value)) => Entry::Set(-value),
                    None => typed("-".to_owned(), &parse),
                });
            }
            Key::Character(c) if c.is_ascii_digit() || *c == '.' => {
                let mut text = match entry.take() {
                    Some(Entry::Typed { text, .. }) => text,
                    // Typing over a set value starts it afresh.
                    _ => String::new(),
                };
                if *c != '.' || !text.contains('.') {
                    text.push(*c);
                }
                *entry = Some(typed(text, &parse));
            }
            _ => return false,
        }
        true
    }
}

fn typed(text: String, parse: impl Fn(&str) -> Option<Real>) -> Entry {
    let value = parse(&text).unwrap_or(0.0);
    Entry::Typed { text, value }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: Real = 1e-9;

    fn number(text: &str) -> Option<Real> {
        text.parse().ok()
    }

    /// Types `keys` into one field, `<` standing for Backspace.
    fn type_into(values: &mut TypedValues, keys: &str) {
        for c in keys.chars() {
            let key = match c {
                '<' => Key::Named(NamedKey::Backspace),
                c => Key::Character(c),
            };
            assert!(values.type_key(&key, false, 3, number), "{c:?} should type");
        }
    }

    fn typed_as(keys: &str) -> TypedValues {
        let mut values = TypedValues::default();
        type_into(&mut values, keys);
        values
    }

    #[test]
    fn digits_and_a_point_build_the_value() {
        let values = typed_as("12.5");
        assert_eq!(values.text(0), Some("12.5"));
        assert!((values.value(0).unwrap() - 12.5).abs() < EPSILON);
    }

    #[test]
    fn a_second_point_is_ignored() {
        assert_eq!(typed_as("1.2.3").text(0), Some("1.23"));
    }

    #[test]
    fn minus_toggles_the_sign_wherever_it_is_typed() {
        assert_eq!(typed_as("-90").value(0), Some(-90.0));
        assert_eq!(typed_as("90-").value(0), Some(-90.0));
        assert_eq!(typed_as("90--").text(0), Some("90"));
    }

    /// Mid-typing, the text reads as nothing yet and counts as zero.
    #[test]
    fn unfinished_text_counts_as_zero() {
        for keys in ["-", ".", "-."] {
            assert_eq!(typed_as(keys).value(0), Some(0.0), "{keys:?}");
        }
    }

    #[test]
    fn backspace_deletes_and_an_emptied_field_is_untyped() {
        let mut values = typed_as("42<");
        assert_eq!(values.text(0), Some("4"));
        type_into(&mut values, "<");
        assert_eq!(values, TypedValues::default());
    }

    #[test]
    fn tab_steps_through_the_fields_and_wraps() {
        let mut values = TypedValues::default();
        let tab = Key::Named(NamedKey::Tab);
        type_into(&mut values, "1");
        values.type_key(&tab, false, 2, number);
        type_into(&mut values, "2");
        values.type_key(&tab, false, 2, number);
        assert_eq!(values.active(2), 0);
        values.type_key(&tab, true, 2, number);
        assert_eq!(values.active(2), 1);
        assert_eq!((values.value(0), values.value(1)), (Some(1.0), Some(2.0)));
    }

    #[test]
    fn the_active_field_stays_within_fewer_fields() {
        let mut values = TypedValues::default();
        let tab = Key::Named(NamedKey::Tab);
        values.type_key(&tab, false, 3, number);
        values.type_key(&tab, false, 3, number);
        assert_eq!(values.active(1), 0);
    }

    #[test]
    fn a_set_value_is_negated_by_minus_and_replaced_by_a_digit() {
        let mut values = TypedValues::default();
        values.set(0, 45.0);
        assert_eq!(values.text(0), None);
        type_into(&mut values, "-");
        assert_eq!(values.value(0), Some(-45.0));
        type_into(&mut values, "7");
        assert_eq!((values.text(0), values.value(0)), (Some("7"), Some(7.0)));
    }

    #[test]
    fn other_keys_do_not_type() {
        let mut values = TypedValues::default();
        for key in [Key::Character('x'), Key::Named(NamedKey::Enter)] {
            assert!(!values.type_key(&key, false, 3, number));
        }
        assert_eq!(values, TypedValues::default());
        assert!(!TypedValues::starts_value(&Key::Named(NamedKey::Tab)));
        assert!(TypedValues::starts_value(&Key::Character('-')));
    }
}
