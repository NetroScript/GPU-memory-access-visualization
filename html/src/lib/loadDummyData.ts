import data from './../../data/basic.json';

export const dummyData = import.meta.env.DEV ? JSON.stringify(data) : '';
