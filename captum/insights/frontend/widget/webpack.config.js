var path = require('path');
var version = require('../package.json').version;

// Custom webpack rules are generally the same for all webpack bundles, hence
// stored in a separate local variable.
var rules = [
    { test: /\.module.css$/, use: [
      'style-loader',
        {
          loader: 'css-loader',
          options: {
            modules: true
          }
        },
      ]
    },
    { test: /^((?!\.module).)*.css$/, use: ['style-loader', 'css-loader'] },
    {
      test: /\.js$/,
      exclude: /node_modules/,
      loaders: 'babel-loader',
      options: {
         presets: ['@babel/preset-react', '@babel/preset-env'],
         plugins: [
            "@babel/plugin-proposal-class-properties"
         ],
      },
    }
]


module.exports = [
    {// Notebook extension
     //
     // This bundle only contains the part of the JavaScript that is run on
     // load of the notebook. This section generally only performs
     // some configuration for requirejs, and provides the legacy
     // "load_ipython_extension" function which is required for any notebook
     // extension.
     //
        mode: 'production',
        entry: './src/extension.js',
        output: {
            filename: 'extension.js',
            path: path.resolve(__dirname, '..', '..', 'widget', 'static'),
            libraryTarget: 'amd'
        },
        resolveLoader: {
          modules: ['../node_modules']
        },
        resolve: {
          modules: ['../node_modules']
        },
    },
    {// Bundle for the notebook containing the custom widget views and models
     //
     // This bundle contains the implementation for the custom widget views and
     // custom widget.
     // It must be an amd module
     //
        mode: 'production',
        entry: './src/index.js',
        output: {
            filename: 'index.js',
            path: path.resolve(__dirname, '..', '..', 'widget', 'static'),
            libraryTarget: 'amd'
        },
        devtool: 'source-map',
        module: {
            rules: rules,
        },
        resolveLoader: {
          modules: ['../node_modules']
        },
        resolve: {
          modules: ['../node_modules']
        },
        externals: ['@jupyter-widgets/base']
    },
    {// Embeddable jupyter-captum-insights bundle
     //
     // This bundle is generally almost identical to the notebook bundle
     // containing the custom widget views and models.
     //
     // The only difference is in the configuration of the webpack public path
     // for the static assets.
     //
     // It will be automatically distributed by unpkg to work with the static
     // widget embedder.
     //
     // The target bundle is always `dist/index.js`, which is the path required
     // by the custom widget embedder.
     //
        mode: 'production',
        entry: './src/embed.js',
        output: {
            filename: 'index.js',
            path: path.resolve(__dirname, '..', '..', 'widget', 'dist'),
            libraryTarget: 'amd',
            publicPath: 'https://unpkg.com/jupyter-captum-insights@' + version + '/dist/'
        },
        devtool: 'source-map',
        module: {
            rules: rules
        },
        resolveLoader: {
          modules: ['../node_modules']
        },
        resolve: {
          modules: ['../node_modules']
        },
        externals: ['@jupyter-widgets/base']
    }
];
