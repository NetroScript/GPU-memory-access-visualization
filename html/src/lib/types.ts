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
  MemoryAccessLog: SingleAccess[];
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


  constructor(address: string,threadId: number, blockId: number,  isRead: boolean, kernelParameters: GenericInformation) {
    this.address = address;
    this.blockIdGlobal = blockId;
    this.isRead = isRead;
    this.threadIdGlobal = threadId;
    this.addressInteger = BigInt("0x" + this.address);

    this.threadID = {
      x: this.threadIdGlobal % kernelParameters.GridDimensions.x,
      y: Math.floor(this.threadIdGlobal / kernelParameters.GridDimensions.x) % kernelParameters.GridDimensions.y,
      z: Math.floor(this.threadIdGlobal / (kernelParameters.GridDimensions.x * kernelParameters.GridDimensions.y))
    }

    this.blockID = {
      x: this.blockIdGlobal % kernelParameters.GridDimensions.x,
      y: Math.floor(this.blockIdGlobal / kernelParameters.GridDimensions.x) % kernelParameters.GridDimensions.y,
      z: Math.floor(this.blockIdGlobal / (kernelParameters.GridDimensions.x * kernelParameters.GridDimensions.y))
    }

    this.kernelParameters = kernelParameters;
  }

}


