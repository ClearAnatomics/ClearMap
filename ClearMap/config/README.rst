config
======

This package manages ClearMap's configuration system: loading, merging,
validating, and persisting YAML/CFG/JSON configuration files.

------------------------------------------------------------------------------------------------------

Composition
===========

config_handler
    Core resolution and IO for configuration files.
    Maps logical section names (``'sample'``, ``'cell_map'``, …) to
    filesystem paths, supporting alternative file names, version-aware
    folder layouts, and multiple formats (YAML, ConfigObj, JSON).
    Also defines ``ConfigProxy`` (a dict with ``write()`` / ``reload()``),
    ``ConfigAlternativesRegistry`` (canonical section names and aliases),
    and the format-specific reader/writer functions.

config_coordinator
    Central in-memory working model for the whole application config.
    Provides ``submit_patch()`` / ``submit()`` as the single write path
    (apply → adjust → validate → commit), thread-safe via ``RLock``.
    Publishes ``CfgChanged`` and ``ChannelsChanged`` events on the bus
    after every committed write.

config_repository
    File I/O and atomic commit layer.  Translates logical section names
    to paths via ``ConfigHandler``, loads all known sections, and writes
    atomically (temp file → rename).  Also handles cloning from a
    template directory and copying from packaged defaults.

defaults_provider
    Loads per-section default configs with a three-level precedence:
    user YAML (``~/.clearmap/defaults/``), packaged YAML, code fallback.
    Optionally validates defaults against JSON schemas.

config_adjusters/
    Adjuster engine that derives dependent config fields from the current
    working model (e.g. inferring registration channels from sample
    config...).  Adjusters are pure and
    idempotent; called automatically by ``ConfigCoordinator`` after each
    patch.

validators/
    JSON-schema-based validation for each config section.
    Schemas live in ``config/schemas/v3.1/``.  ``validate_all()`` is called
    by ``ConfigCoordinator.validate()`` before every commit.

update_config
    Migration utilities to upgrade config files from older ClearMap
    versions to the current layout.

atlas
    Atlas name/ID/resolution registry used by the registration UI.

compound_keys
    ``PairKey`` — a canonical ordered-pair key for colocalization channel
    pairs.

change_detection
    Helpers that inspect a set of changed config keys (from a
    ``CfgChanged`` event) to determine whether tabs need rebuilding or
    workers need reconciliation (e.g. ``channel_data_type_changed``,
    ``channels_added_or_removed``).

defaults/v3.1/
    Packaged YAML default configs for each pipeline section, installed
    alongside the package.  Copied into new experiment directories on
    first open.

machine_params / display_params
    Global (non-experiment) configs for hardware preferences and display
    defaults.  Stored in ``~/.clearmap/`` and edited via the Preferences
    dialog.