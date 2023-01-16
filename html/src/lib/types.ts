export interface GenericInformation {
  GridDimensions: Dimension;
  BlockDimensions: Dimension;
  WarpSize: number;
}

export interface MemoryRegions {
  StartAddress: string;
  EndAddress: string;
  NumberOfElements: number;
  SizeOfSingleElement: number;
  Name: string;
}

// A Single Access describes the address, the blockID, the threadID and whether it is a read or write access (read = true)
export type SingleAccess = [string, number, number, boolean]

export interface Dimension {
  x: number;
  y: number;
  z: number;
}

// This describes the information we get from a template output file
export interface OutputJSON {
  GlobalSettings: GenericInformation;
  MemoryRegions: MemoryRegions[];
  MemoryAccessLogs: SingleAccess[];
}

export class AccessInstance {

  readonly addressInteger: bigint;
  readonly address: string;
  readonly blockIdGlobal: number;
  readonly isRead: boolean;
  readonly threadIdGlobal: number;

  readonly threadID: Dimension;
  readonly blockID: Dimension;

  readonly kernelParameters: GenericInformation;

  public index: number;


  constructor(address: string, threadId: number, blockId: number, isRead: boolean, kernelParameters: GenericInformation) {
    this.address = address;
    this.blockIdGlobal = blockId;
    this.isRead = isRead;
    this.threadIdGlobal = threadId;
    this.addressInteger = BigInt(this.address);

    this.threadID = {
      x: this.threadIdGlobal % kernelParameters.GridDimensions.x,
      y: Math.floor(this.threadIdGlobal / kernelParameters.GridDimensions.x) % kernelParameters.GridDimensions.y,
      z: Math.floor(this.threadIdGlobal / (kernelParameters.GridDimensions.x * kernelParameters.GridDimensions.y))
    };

    this.blockID = {
      x: this.blockIdGlobal % kernelParameters.GridDimensions.x,
      y: Math.floor(this.blockIdGlobal / kernelParameters.GridDimensions.x) % kernelParameters.GridDimensions.y,
      z: Math.floor(this.blockIdGlobal / (kernelParameters.GridDimensions.x * kernelParameters.GridDimensions.y))
    };

    this.kernelParameters = kernelParameters;
  }

}

interface MemoryRegionManagerDisplaySettings {
  width: number;
  height: number;
  isMultiDimensional: boolean;
  using1DSparseRepresentation: boolean;
}

export class MemoryRegionManager {

  readonly startAddress: string;
  readonly startAddressInteger: bigint;
  readonly endAddress: string;
  readonly endAddressInteger: bigint;
  readonly numberOfElements: number;
  readonly sizeOfSingleElement: number;
  readonly name: string;
  readonly memoryAccesses: AccessInstance[] = [];

  // For each index of the array data structure, we store all accesses which read from this index
  readonly readAccesses: Map<number, AccessInstance[]>
  // For each index of the array data structure, we store all accesses which write to this index
  readonly writeAccesses: Map<number, AccessInstance[]>


  // Store display Settings
  displaySettings: MemoryRegionManagerDisplaySettings  = {width: -1, height: 0, isMultiDimensional: false, using1DSparseRepresentation: true};

  private lowestIndex: number = Number.MAX_SAFE_INTEGER;
  private highestIndex: number = Number.MIN_SAFE_INTEGER;

  // Store the maximum number of accesses in a single index to later represent it in a heatmap
  private highestReadCount: number = 0;
  private highestWriteCount: number = 0;

  // For 1d arrays, we can display a sparse representation which only shows accessed addresses and not the entire memory space
  // if it is a normal number, it is the valid index of the array, if it is a tuple of numbers, this represents the empty spaces between memory accesses
  private sparse1DMemoryLocations: (number | [number, number])[] = [];

  // Getter for the length of all elements in the memory region
  get length() {
    return this.memoryAccesses.length;
  }

  // Getter for the lowest index of the memory region
  getLowestIndex() {
    return this.lowestIndex;
  }

  // Getter for the highest index of the memory region
  getHighestIndex() {
    return this.highestIndex;
  }

  constructor(memoryRegion: MemoryRegions) {
    this.startAddress = memoryRegion.StartAddress;
    this.startAddressInteger = BigInt(this.startAddress);
    this.endAddress = memoryRegion.EndAddress;
    this.endAddressInteger = BigInt(this.endAddress);
    this.numberOfElements = memoryRegion.NumberOfElements;
    this.sizeOfSingleElement = memoryRegion.SizeOfSingleElement;
    this.name = memoryRegion.Name;

    this.readAccesses = new Map<number, AccessInstance[]>();
    this.writeAccesses = new Map<number, AccessInstance[]>();
  }

  addMemoryAccess(access: AccessInstance) {
    // Get the index of the access to set it on the access
    access.index = this.convertAccessToIndex(access);

    // Check if the index changes the boundaries of the memory region
    if (access.index < this.lowestIndex) {
      this.lowestIndex = access.index;
    }
    if (access.index > this.highestIndex) {
      this.highestIndex = access.index;
    }

    // Store the access in the memoryAccesses array
    this.memoryAccesses.push(access);

    // Store the access in the read or write map and create the array if it does not exist
    if (access.isRead) {
      if (!this.readAccesses.has(access.index)) {
        this.readAccesses.set(access.index, []);
      }
      this.readAccesses.get(access.index)?.push(access);

      // Check if the number of accesses at this index for reading is the highest so far
      if (this.readAccesses.get(access.index)?.length > this.highestReadCount) {
        this.highestReadCount = this.readAccesses.get(access.index)?.length || 0;
      }
    }

    if (!access.isRead) {
      if (!this.writeAccesses.has(access.index)) {
        this.writeAccesses.set(access.index, []);
      }
      this.writeAccesses.get(access.index)?.push(access);

      // Check if the number of accesses at this index for writing is the highest so far
      if (this.writeAccesses.get(access.index)?.length > this.highestWriteCount) {
        this.highestWriteCount = this.writeAccesses.get(access.index)?.length || 0;
      }
    }
  }

  addMemoryAccessIfInRegion(access: AccessInstance) {
    if (this.checkIfAccessIsInRegion(access)) {
      this.addMemoryAccess(access);
    }
  }


  checkIfAddressIsInRegion(address: bigint) : boolean {
    return address >= this.startAddressInteger && address < this.endAddressInteger;
  }

  checkIfAddressStringIsInRegion(address: string) : boolean {
    return this.checkIfAddressIsInRegion(BigInt(address));
  }

  checkIfAccessIsInRegion(access: AccessInstance) : boolean {
    return this.checkIfAddressIsInRegion(access.addressInteger);
  }

  convertAddressToIndex(address: bigint) : number {
    return Number((address - this.startAddressInteger)/BigInt(this.sizeOfSingleElement));
  }

  convertAddressStringToIndex(address: string) : number {
    return this.convertAddressToIndex(BigInt(address));
  }

  convertAccessToIndex(access: AccessInstance) : number {
    return this.convertAddressToIndex(access.addressInteger);
  }

  convertIndexToAddress(index: number) : bigint {
    return this.startAddressInteger + BigInt(index * this.sizeOfSingleElement);
  }

  // Return a 0x hex string padded to a width of 64 bits (16 hex characters)
  convertIndexToAddressString(index: number) : string {
    return "0x" + this.convertIndexToAddress(index).toString(16).padStart(16, "0");
  }

  // Getter for the read accesses of a specific index
  getReadAccesses(index: number) : AccessInstance[] {
    return this.readAccesses.get(index) || [];
  }

  // Getter for the write accesses of a specific index
  getWriteAccesses(index: number) : AccessInstance[]  {
    return this.writeAccesses.get(index) || [];
  }

  // Getter for the read accesses of specific address
  getReadAccessesOfAddress(address: bigint) : AccessInstance[]  {
    return this.readAccesses.get(this.convertAddressToIndex(address)) || [];
  }

  // Getter for the write accesses of specific address
  getWriteAccessesOfAddress(address: bigint) : AccessInstance[] {
    return this.writeAccesses.get(this.convertAddressToIndex(address)) || [];
  }

  // Getter for all accesses of a specific index
  getAllAccesses(index: number) : AccessInstance[] {
    return [...(this.readAccesses.get(index) || []), ...(this.writeAccesses.get(index) || [])];
  }

  // Getter for all accesses of a specific address
  getAllAccessesOfAddress(address: bigint) : AccessInstance[] {
    return this.getAllAccesses(this.convertAddressToIndex(address));
  }

  // Getter for sparse 1d memory locations
  getSparse1DMemoryLocations() : (number | [number, number])[] {
    return this.sparse1DMemoryLocations;
  }

  // This function generates all static data after all memory accesses were added to the memory region
  bake() : void {
    // Right now this function just constructs the sparse representation of the memory region

    // Clear the sparse1DMemoryLocations array
    this.sparse1DMemoryLocations.length = 0;

    // For that we first need to merge the indexes for both the read and write maps
    let allIndexes = [...this.readAccesses.keys(), ...this.writeAccesses.keys()];
    // Then we need to remove duplicates
    allIndexes = [...new Set(allIndexes)];
    // And sort the indexes
    allIndexes.sort((a, b) => a - b);

    // Now we can construct the sparse representation
    // For that we iterate in order over all indexes, and if there is a gap greater than 1, we add a sparse element
    for (let i = 0; i < allIndexes.length; i++){
      let index = allIndexes[i];
      // Also store the next index if it exists
      let nextIndex = allIndexes[i+1] || index;

      // Always add existing indexes
      this.sparse1DMemoryLocations.push(index)

      // If the next index is greater than the current index by more than 1, we need to add a sparse element which represents the gap
      if (nextIndex > index + 1) {
        this.sparse1DMemoryLocations.push([index + 1, nextIndex-1])
      }
    }
  }
}
