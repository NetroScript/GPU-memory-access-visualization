import { derived, writable } from 'svelte/store';
import type { MemoryRegionManager } from './types';

// Implement a store storing the state of the drawer information

export interface DrawerState {
  currentMemoryRegion: MemoryRegionManager;
  currentMemoryRegionIndex: number;
  showSingleAccessTable: boolean;
  showReadAccess: boolean;
}

export interface PageState {
  backGroundContrastBlack: boolean;
  showIndex: boolean;
  showCombinedAccess: boolean;
  availableMemoryRegions: MemoryRegionManager[];
  showGrid: boolean;
  customMemoryWidth: number;
}

export const ListPlaceHolder = {
  // eslint-disable-next-line no-prototype-builtins,@typescript-eslint/ban-types
  includes: (object: {}) => object.hasOwnProperty('isPlaceHolder'),
  isPlaceHolder: true
};

export const drawerState = writable<DrawerState>({
  currentMemoryRegion: null,
  currentMemoryRegionIndex: -1,
  showSingleAccessTable: false,
  showReadAccess: true
});
export const drawerContent = derived(drawerState, ($drawerState) => {
  if ($drawerState.currentMemoryRegion === null) {
    return [];
  }

  if ($drawerState.showSingleAccessTable) {
    return $drawerState.currentMemoryRegion.getAllAccesses($drawerState.currentMemoryRegionIndex);
  }
  if ($drawerState.showReadAccess) {
    return $drawerState.currentMemoryRegion.getReadAccesses($drawerState.currentMemoryRegionIndex);
  } else {
    return $drawerState.currentMemoryRegion.getWriteAccesses($drawerState.currentMemoryRegionIndex);
  }
});

export const pageState = writable<PageState>({
  backGroundContrastBlack: true,
  showIndex: false,
  availableMemoryRegions: [],
  showGrid: true,
  customMemoryWidth: 0,
  showCombinedAccess: false
});

export const currentMemoryRegion = writable<MemoryRegionManager | typeof ListPlaceHolder>(ListPlaceHolder);
