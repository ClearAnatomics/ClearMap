from dataclasses import dataclass
from typing import Tuple, List, Any, Optional, Dict

# Intent may not actually materialize in the config
@dataclass(frozen=True)
class UiChannelRenamed:
    old: str
    new: str

@dataclass(frozen=True)
class UiChannelsChanged:
    before: List[str]
    after: List[str]

# Fact: actual change in the config
@dataclass(frozen=True)
class ChannelRenamed:
    old: str
    new: str

@dataclass(frozen=True)
class ChannelsChanged:
    before: List[str]
    after: List[str]

@dataclass(frozen=True)
class ChannelsSnapshot:
    names: List[str]

@dataclass(frozen=True)
class ChannelDefaultsChanged:
    partners: Dict[str, Dict[str, Optional[str]]]

@dataclass(frozen=True)
class WorkspaceChanged:
    exp_dir: str


@dataclass(frozen=True)
class UiOrientationChanged:
    channel_name: str
    orientation: Tuple[int, int, int]

@dataclass(frozen=True)
class UiCropChanged:
    channel_name: str
    slice_x: Any
    slice_y: Any
    slice_z: Any

@dataclass(frozen=True)
class UiConvertToClearMapFormat:
    channel_name: str

@dataclass(frozen=True)
class UiRequestPlotMiniBrain:
    channel_index: int

@dataclass(frozen=True)
class UiRequestPlotAtlas:
    channel_index: int

@dataclass(frozen=True)
class UiRequestLandmarksDialog:
    page_index: int


@dataclass(frozen=True)
class UiLayoutChannelChanged:
    channel_name: str
    layout_channel: str

@dataclass(frozen=True)
class UiUseExistingLayoutChanged:
    channel_name: str
    use_existing: bool

@dataclass(frozen=True)
class UiAlignWithChanged:
    channel_name: str
    align_with: Optional[str]

@dataclass(frozen=True)
class UiAtlasIdChanged:
    atlas_base_name: str

@dataclass(frozen=True)
class UiAtlasStructureTreeIdChanged:
    tree_id: str

@dataclass(frozen=True)
class UiVesselGraphFiltersChanged:
    pass


@dataclass(frozen=True)
class TabsUpdated:
    titles: List[str]
    tabs: List[Any]

@dataclass(frozen=True)
class UiTabActivated:
    key: str

@dataclass(frozen=True)
class TabActivationResult:
    key: str
    ok: bool
    message: str


@dataclass(frozen=True)
class UiRequestRefreshTabs:
    pass

# App/domain → UI notifications
@dataclass(frozen=True)
class CfgChanged:
    changed_keys: Tuple[str, ...]


@dataclass(frozen=True)
class RegistrationStatusChanged:
    pass


@dataclass(frozen=True)
class UiBatchResultsFolderChanged:
    folder: str

@dataclass(frozen=True)
class UiBatchGroupsChanged:
    groups: Dict[str, List[str]]
