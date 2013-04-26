define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'views/tree'
], function($, _, Backbone, tree) {
    /*
     `TreeControlView` is a button bar that sits above the tree view and allows
     the user to add, edit and remove settings values, movers and spills using
     button clicks.
     */
    var TreeControlView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            this.addButtonEl = this.options.addButtonEl;
            this.removeButtonEl = this.options.removeButtonEl;
            this.settingsButtonEl = this.options.settingsButtonEl;

            // Controls that require the user to select an item in the TreeView.
            this.itemControls = [this.removeButtonEl, this.settingsButtonEl];
            this.disableControls();
            this.setupClickEvents();

            this.options.treeView.on(tree.TreeView.ITEM_ACTIVATED, this.treeItemActivated);
        },

        setupClickEvents: function() {
            var _this = this;
            var clickEvents = [
                [this.addButtonEl, TreeControlView.ADD_BUTTON_CLICKED],
                [this.removeButtonEl, TreeControlView.REMOVE_BUTTON_CLICKED],
                [this.settingsButtonEl, TreeControlView.SETTINGS_BUTTON_CLICKED]
            ];

            _.each(_.object(clickEvents), function(customEvent, element) {
                $(element).click(function(event) {
                    if ($(_this).hasClass('disabled')) {
                        return false;
                    }
                    _this.trigger(customEvent);
                    return true;
                });
            });
        },

        treeItemActivated: function() {
            this.enableControls();
        },

        enableControls: function() {
            _.each(this.itemControls, function(buttonEl) {
                $(buttonEl).removeClass('disabled');
            });
        },

        disableControls: function() {
            _.each(this.itemControls, function(buttonEl) {
                $(buttonEl).addClass('disabled');
            });
        }
    }, {
        // Event constants
        ADD_BUTTON_CLICKED: 'gnome:addItemButtonClicked',
        REMOVE_BUTTON_CLICKED: 'gnome:removeItemButtonClicked',
        SETTINGS_BUTTON_CLICKED: 'gnome:itemSettingsButtonClicked'
    });

    return {
        TreeControlView: TreeControlView
    }
});