define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'lib/jquery.dynatree',
], function($, _, Backbone) {

    /*
     `TreeView` is a representation of the user's current model displayed as a tree
     of items that the user may click or double-click on to display add/edit forms
     for model settings, movers and spills.
     */
    var TreeView = Backbone.View.extend({
        initialize: function() {
            var _this = this;
            _.bindAll(this);
            this.treeEl = this.options.treeEl;
            this.gnomeModel = this.options.gnomeModel;
            this.url = this.gnomeModel.url() + "/tree";

            // Turn off node icons. A [+] icon will still appear for nodes
            // that have children.
            $.ui.dynatree.nodedatadefaults["icon"] = false;
            this.tree = this.setupDynatree();

            _.each(this.options.collections, function(collection) {
                collection.on('sync', _this.reload);
                collection.on('add', _this.reload);
                collection.on('destroy', _this.reload);
            });

            this.gnomeModel.on('sync', this.reload);
            this.options.map.on('sync', this.reload);
            this.options.map.on('destroy', this.reload);

            $(window).bind('resize', function() {
                _this.resize();
            });

            _this.resize();
        },

        /*
         Adjust the sidebar height to stay at 100% of the page minus the navbar.
         */
        resize: function() {
            var windowHeight = $(window).height();
            var navbarBorderRadius = 4;
            var navbarHeight = $('.navbar').height() + navbarBorderRadius;
            var newSidebarHeight = windowHeight - navbarHeight;

            // the sidebar-container inside the sidebar has
            // top padding.  I can't see a way to query this,
            // so we just add a static value.
            var sbContainerPadding = 14;

            // the btn-toolbar inside the sidebar exists, but has
            // a height of 0 when first rendering the page
            if ($('#sidebar div.btn-toolbar').height() > 0) {
            	var sbButtonGroupHeight = $('#sidebar div.btn-toolbar').height();
            }
            else {
            	var sbButtonGroupHeight = 34;
            }

            var tree = $('#tree');
            var sidebar = $('#sidebar');
            var treeHeight = tree.height();
            var treeHeightDiff = sidebar.height() - treeHeight;
            sidebar.height(newSidebarHeight);
            tree.height(newSidebarHeight - sbButtonGroupHeight - sbContainerPadding);
        },

        setupDynatree: function() {
            var _this = this;

            return $(this.treeEl).dynatree({
                onActivate: function(node) {
                    _this.trigger(TreeView.ITEM_ACTIVATED, node);
                },
                onPostInit: function(isReloading, isError) {
                    // Fire events for a tree that was reloaded from cookies.
                    // isReloading is true if status was read from existing cookies.
                    // isError is only used in Ajax mode
                    this.reactivate();

                     // Expand all items
                    this.getRoot().visit(function (node) {
                        node.expand(true);
                    });
                },
                onDblClick: function(node, event) {
                    _this.trigger(TreeView.ITEM_DOUBLE_CLICKED, node);
                },
                initAjax: {
                    url: _this.url
                },
                persist: true
            });
        },

        getActiveItem: function() {
            return this.tree.dynatree("getActiveNode");
        },

        reload: function(model, attrs, opts) {
            var _this = this;
            if (opts.reloadTree === false) {
                return;
            }
            if (this.gnomeModel && this.gnomeModel.wasDeleted) {
                return;
            }
            this.tree.dynatree('getTree').reload(function () {
                _this.trigger(TreeView.RELOADED);
                _this.resize();
            });
        }
    }, {
        ITEM_ACTIVATED: 'treeView:treeItemActivated',
        ITEM_DOUBLE_CLICKED: 'treeView:treeItemDoubleClicked',
        RELOADED: 'treeView:reloaded'
    });

    return {
        TreeView: TreeView
    }
});