var path = require("path");

// Custom webpack rules are generally the same for all webpack bundles, hence
// stored in a separate local variable.
var rules = [
  {
    test: /\.module.css$/,
    use: [
      "style-loader",
      {
        loader: "css-loader",
        options: {
          modules: true,
        },
      },
    ],
  },
  { test: /^((?!\.module).)*.css$/, use: ["style-loader", "css-loader"] },
  {
    test: /\.(js|ts|tsx)$/,
    exclude: /node_modules/,
    loaders: "babel-loader",
    options: {
      presets: [
        "@babel/preset-react",
        "@babel/preset-env",
        "@babel/preset-typescript",
      ],
      plugins: ["@babel/plugin-proposal-class-properties"],
    },
  },
];

var extensions = [".js", ".ts", ".tsx"];

module.exports = [
  {
    // Notebook extension
    //
    // This bundle only contains the part of the JavaScript that is run on
    // load of the notebook. This section generally only performs
    // some configuration for requirejs, and provides the legacy
    // "load_ipython_extension" function which is required for any notebook
    // extension.
    //
    mode: "production",
    entry: "./src/extension.js",
    output: {
      filename: "extension.js",
      path: path.resolve(__dirname, "..", "..", "widget", "static"),
      libraryTarget: "amd",
    },
    resolveLoader: {
      modules: ["../node_modules"],
      extensions: extensions,
    },
    resolve: {
      modules: ["../node_modules"],
    },
    externals: ["moment"], // Removes unused dependency-of-dependency
  },
  {
    // Bundle for the notebook containing the custom widget views and models
    //
    // This bundle contains the implementation for the custom widget views and
    // custom widget.
    // It must be an amd module
    //
    mode: "production",
    entry: "./src/index.js",
    output: {
      filename: "index.js",
      path: path.resolve(__dirname, "..", "..", "widget", "static"),
      libraryTarget: "amd",
    },
    module: {
      rules: rules,
    },
    resolveLoader: {
      modules: ["../node_modules"],
    },
    resolve: {
      modules: ["../node_modules"],
      extensions: extensions,
    },
    externals: ["@jupyter-widgets/base", "moment"],
  },
];
