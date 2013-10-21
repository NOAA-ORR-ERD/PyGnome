define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    '//netdna.bootstrapcdn.com/bootstrap/3.0.0/js/bootstrap.min.js',
 ], function($, _, Backbone) {
    /*
     `MenuView` handles the drop-down menus on the top of the page. The object
     listens for click events on menu items and fires specialized events, like
     RUN_ITEM_CLICKED, which an `AppView` object listens for.

     Most of these functions exist elsewhere in the application and `AppView`
     calls the appropriate method for whatever functionality the user invoked.
     */
    var MenuView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);
            // Top-level drop-downs
            this.modelDropdownEl = this.options.modelDropdownEl;
            this.runDropdownEl = this.options.runDropdownEl;
            this.helpDropdownEl = this.options.helpDropdownEl;

            // Drop-down children
            this.newItemEl = this.options.newItemEl;
            this.runItemEl = this.options.runItemEl;
            this.stepItemEl = this.options.stepItemEl;
            this.runUntilItemEl = this.options.runUntilItemEl;
            this.longIslandItemEl = this.options.longIslandItemEl;

            $(this.newItemEl).click(this.newItemClicked);
            $(this.runItemEl).click(this.runItemClicked);
            $(this.runUntilItemEl).click(this.runUntilItemClicked);

            $('ul.nav').on('click', '.location-file-item', this.locationFileItemClicked);
        },

        hideDropdown: function() {
            $(this.modelDropdownEl).dropdown('toggle');
        },

        newItemClicked: function(event) {
            this.hideDropdown();
            this.trigger(MenuView.NEW_ITEM_CLICKED);
        },

        runItemClicked: function(event) {
            this.hideDropdown();
            this.trigger(MenuView.RUN_ITEM_CLICKED);
        },

        runUntilItemClicked: function(event) {
            this.hideDropdown();
            this.trigger(MenuView.RUN_UNTIL_ITEM_CLICKED);
        },

        locationFileItemClicked: function(event) {
            event.preventDefault();
            this.hideDropdown();
            var location = $(event.target).data('location');
            this.trigger(MenuView.LOCATION_FILE_ITEM_CLICKED, location);
        }
    }, {
        // Event constants
        NEW_ITEM_CLICKED: "menuView:newMenuItemClicked",
        RUN_ITEM_CLICKED: "menuView:runMenuItemClicked",
        RUN_UNTIL_ITEM_CLICKED: "menuView:runUntilMenuItemClicked",
        LOCATION_FILE_ITEM_CLICKED: "menuView:locationFileItemClicked"
    });

    return {
        MenuView: MenuView
    };
});