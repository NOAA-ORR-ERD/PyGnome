define([
    'jquery',
    'lib/underscore',
    'views/forms/modal',
    'lib/jquery.ui'
], function($, _, modal) {
    /*
     A base class for modal forms that ask the user to choose from a list of
     object types that are themselves represented by a `FormView` instance.
     */
    var ChooseObjectTypeFormView = modal.JQueryUIModalFormView.extend({
        initialize: function() {
            var _this = this;

            this.options.dialog = _.extend({
                height: 175,
                width: 400,
                buttons: {
                    Cancel: function() {
                        _this.cancel();
                        $(this).dialog("close");
                    },

                    Choose: function() {
                        _this.submit();
                        $(this).dialog("close");
                    }
                }
            }, this.options.dialog);

            ChooseObjectTypeFormView.__super__.initialize.apply(this, arguments);
        }
    });


    /*
     This is a non-AJAX-enabled modal form object to support the "add mover" form,
     which asks the user to choose a type of mover to add. We then use the selection
     to display another, mover-specific form.
     */
    var AddMoverFormView = ChooseObjectTypeFormView.extend({
        submit: function() {
            var moverType = this.getElementByName('mover-type').val();

            if (moverType) {
                this.trigger(AddMoverFormView.MOVER_CHOSEN, moverType);
                this.hide();
            }

            return false;
        }
    }, {
        // Events
        MOVER_CHOSEN: 'addMoverFormView:moverChosen'
    });


    var AddEnvironmentFormView = ChooseObjectTypeFormView.extend({
        submit: function() {
            var environmentType = this.getElementByName('environment-type').val();

            if (environmentType) {
                this.trigger(AddEnvironmentFormView.ENVIRONMENT_CHOSEN, environmentType);
                this.hide();
            }

            return false;
        }
    }, {
        // Events
        ENVIRONMENT_CHOSEN: 'addEnvironmentFormView:environmentChosen'
    });


    /*
    This is a non-AJAX-enabled modal form object to support the "add spill"
    form, which asks the user to choose a type of spill to add. We then use the
    selection to display another, spill-specific form.
    */
    var AddSpillFormView = ChooseObjectTypeFormView.extend({
        show: function(startCoords, endCoords) {
            this.startCoords = startCoords;
            this.endCoords = endCoords;

            AddSpillFormView.__super__.show.apply(this);
        },

        submit: function() {
            var spillType = this.getElementByName('spill-type').val();

            if (spillType) {
                this.trigger(AddSpillFormView.SPILL_CHOSEN, spillType, this.startCoords, this.endCoords);
                this.coords = null;
                this.hide();
            }

            return false;
        },

        cancel: function() {
            this.trigger(AddSpillFormView.CANCELED, this);
        }
    }, {
        // Event constants
        SPILL_CHOSEN: 'addSpillFormView:spillChosen',
        CANCELED: 'addSpillFormView:canceled'
    });


    var AddMapFormView = ChooseObjectTypeFormView.extend({
        submit: function() {
            var source = this.getElementByName('map-source').val();
            if (source) {
                this.trigger(AddMapFormView.SOURCE_CHOSEN, source);
            }
        }
    }, {
        // Event constants
        SOURCE_CHOSEN: 'addMapFormView:sourceChosen'
    });


    return {
        ChooseObjectTypeFormView: ChooseObjectTypeFormView,
        AddMoverFormView: AddMoverFormView,
        AddEnvironmentFormView: AddEnvironmentFormView,
        AddSpillFormView: AddSpillFormView,
        AddMapFormView: AddMapFormView
    }
});