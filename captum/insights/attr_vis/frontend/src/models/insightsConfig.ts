export enum ArgumentType {
  Number = "number",
  Enum = "enum",
  String = "string",
  Boolean = "boolean",
}

export type GenericArgumentConfig<T> = {
  value: T;
  limit: T[];
};

export type ArgumentConfig =
  | ({ type: ArgumentType.Number } & GenericArgumentConfig<number>)
  | ({ type: ArgumentType.Enum } & GenericArgumentConfig<string>)
  | ({ type: ArgumentType.String } & { value: string })
  | ({ type: ArgumentType.Boolean } & { value: boolean });

export interface MethodsArguments {
  [method_name: string]: {
    [arg_name: string]: ArgumentConfig;
  };
}

export interface InsightsConfig {
  classes: string[];
  methods: string[];
  method_arguments: MethodsArguments;
  selected_method: string;
}
