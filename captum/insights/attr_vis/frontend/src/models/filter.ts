export interface FilterConfig {
  attribution_method: string;
  arguments: { [key: string]: any };
  prediction: string;
  classes: string[];
}

export interface TagClass {
  id: number;
  name: string;
}
