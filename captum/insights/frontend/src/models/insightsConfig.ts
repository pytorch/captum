interface ConfigParameters {
    params: { [key: string]: any };
    help_info?: string;
    post_process?: { [key: string]: (...args: any[]) => any };
}

export interface InsightsConfig {
    classes: string[];
    methods: string[];
    method_arguments: {
        [method_name: string]: ConfigParameters
    }
    selected_method: string;
}