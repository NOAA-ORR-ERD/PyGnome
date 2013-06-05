define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'lib/moment'
], function($, _, Backbone) {
    /*
     A base class for views.

     Includes helper methods for getting elements within the view's `el`
     and constructing moment objects from dates spread across multiple form
     elements.

     Right now this is mostly used only in views/forms prototypes.
     */
    var BaseView = Backbone.View.extend({
        initialize: function() {
            this.model = this.options.model;
            this.id = this.options.id;

            if (this.options.id) {
                this.$el = $('#' + this.options.id);
            }

            BaseView.__super__.initialize.apply(this, arguments);
        },

        /*
         Get an element by searching for its name within `container`.

         If `container` is undefined, default to the FormView's element.
         */
        getElementByName: function(name, container, type) {
            container = container && container.length ? container : this.$el;
            type = type ? type : '*';
            return container.find(type + '[name="' + name + '"]');
        },

        setDateFields: function(target, datetime) {
            if (typeof target === 'string') {
                target = this.$el.find(target);
            }

            if (!datetime || target.length === 0) {
                return;
            }

            var fields = this.getDateFields(target);
            fields.date.val(datetime.format("MM/DD/YYYY"));
            fields.hour.val(datetime.format('HH'));
            fields.minute.val(datetime.format('mm'));
        },

        hasDateFields: function(target) {
            var fields = this.getDateFields(target);
            return fields.date && fields.hour && fields.minute;
        },

        getDateFields: function(target) {
            target = $(target);

            if (target.length === 0) {
                return;
            }

            return {
                date: target.find('.date'),
                hour: target.find('.hour'),
                minute: target.find('.minute')
            }
        },

        getDate: function(target) {
            if (target.length === 0) {
                return;
            }

            var fields = this.getDateFields(target);
            var date = fields.date.val();
            var hour = fields.hour.val();
            var minute = fields.minute.val();

            if (hour && minute) {
                date = date + ' ' + hour + ':' + minute;
            }

            // TODO: Handle a date-parsing error here.
            if (date) {
                return moment(date);
            }
        },

        clearErrors: function() {
            var groups = this.$el.find('.control-group');
            var errors = this.$el.find('a.error');

            if (groups.length) {
                groups.removeClass('error');
            }

            if (errors.length) {
                errors.attr('title', '');
                errors.addClass('hidden');
            }
        },

        sendMessage: function(message) {
            this.trigger(BaseView.MESSAGE_READY, message);
        }
    }, {
        MESSAGE_READY: 'baseView:messageReady',
    });


    function ModelNotFoundException(message) {
        this.message = message;
        this.name = "ModelNotFoundException";
    }

    return {
        BaseView: BaseView,
        ModelNotFoundException: ModelNotFoundException
    }
});
