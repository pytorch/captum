//TODO: should be renamed to ArgumentType
export enum ConfigType {
    Number = "number",
    Enum = "enum",
    String = "string",
    Boolean = "boolean"
}

export type GenericArgumentConfig<T> = {
    value: T;
    limit: T[];
}

export type ArgumentConfig =
    { type: ConfigType.Number } & GenericArgumentConfig<number> |
    { type: ConfigType.Enum } & GenericArgumentConfig<string> |
    { type: ConfigType.String } & { value: string } |
    { type: ConfigType.Boolean } & { value: boolean }

export interface MethodsArguments {
    [method_name: string]: {
        [arg_name: string]: ArgumentConfig;
    }
}

export interface InsightsConfig {
    classes: string[];
    methods: string[];
    method_arguments: MethodsArguments;
    selected_method: string;
}