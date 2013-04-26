define([
    'jquery',
    'lib/underscore',
    'models',
    'util',
    'views/forms/deferreds',
    'views/forms/multi_step',
    'views/forms/base',
], function($, _, models, util, deferreds, multi_step, base) {
    /*
     A multi-step form that shows screens configured for a Gnome Wizard.

     Each "step" in the multi-step form appears as a different modal dialog
     box. Wizards can also reference other `FormView` objects loaded on the
     page, open them, and return to the wizard when the user "submits" the
     referenced form. These are handled as deferred submits -- the forms are
     validated via their validator web service and, if the data validates,
     are added to the `deferreds` queue for submission when the user submits
     the final wizard form.
     */
    var LocationFileWizardFormView = multi_step.MultiStepFormView.extend({
        initialize: function() {
            LocationFileWizardFormView.__super__.initialize.apply(this, arguments);

            _.bindAll(this);

            this.widget = this.$el.dialog('widget');
            this.widget.on('click', '.ui-button.next', this.next);
            this.widget.on('click', '.ui-button.back', this.back);
            this.widget.on('click', '.ui-button.cancel', this.cancel);
            this.widget.on('click', '.ui-button.finish', this.finish);
            this.widget.on('click', '.ui-button.references', this.showReferences);

            this.locationFileMeta = this.options.locationFileMeta;

            this.references = this.$el.find('div.references').dialog({
                autoOpen: false,
                buttons: {
                    Ok: function() {
                        $(this).dialog("close");
                    }
                }
            });

            this.setButtons();
        },

        setStepSize: function() {
            var step = this.getCurrentStep();
            var height = step.data('height');
            var width = step.data('width');

            if (height) {
                this.$el.dialog('option', 'height', parseInt(height));
            }
            if (width) {
                this.$el.dialog('option', 'width', parseInt(width));
            }
        },

        getDataBindings: function() {
            return {wizard: this.model};
        },

        resetModel: function() {
            // Don't reset the model for this form
        },

        /*
         Finish the wizard. In order:

            - Load the location file parameters into the user's current model
            - Run 'deferred' functions -- model-related form submits
            - Save the location-file-specific parameters the user selected
            - Reload the page
         */
        finish: function() {
            var _this = this;
             deferreds.run().done(function() {
                 _this.model.save().done(function() {
                     // TODO: Trigger event, let AppView handle this.
                     util.refresh();
                 }).fail(function() {
                     console.log('Error submitting the location file wizard.');
                     alert('Error setting up your model. Please try again.');
                 });
             }).fail(function() {
                 console.log('Error running deferred methods.');
                 alert('Error setting up your model. Please try again.');
             });

        },

        loadLocationFile: function() {
            var model = new models.GnomeModelFromLocationFile({
                location_name: this.locationFileMeta.get('filename')
            }, {
                gnomeModel: this.gnomeModel
            });

            return model.save();
        },

        showReferences: function() {
            this.references.dialog('open');
        },

        setButtons: function() {
            var step = this.getCurrentStep();
            var buttons = step.find('.custom-dialog-buttons');
            if (buttons.length) {
                var buttonPane = this.widget.find('.ui-dialog-buttonpane');
                buttonPane.empty();
                buttons.clone().appendTo(buttonPane).removeClass('hidden');
            }
        },

        getCustomStepButtons: function(step) {
            var buttons = [];
            var customFormButtons = step.find('.custom-dialog-buttons').find(
                '.ui-button');

            if (customFormButtons.length) {
                _.each(customFormButtons, function(el, index, list) {
                    var fnName = $(el).data('function-name');
                    if (!fnName) {
                        return;
                    }
                    buttons.push({
                        deferred: $(el).data('deferred'),
                        text: $(el).text(),
                        fnName: fnName
                    });
                })
            }

            return buttons;
        },

        showStep: function(step) {
            LocationFileWizardFormView.__super__.showStep.apply(this, [step]);
            this.setButtons();

            var form = step.data('show-form');
            if (form) {
                this.widget.addClass('hidden');
                this.showForm(form, this.getCustomStepButtons(step));
            } else {
                this.widget.removeClass('hidden');
                this.setStepSize();
            }
        },

        showForm: function(form, customButtons) {
            var _this = this;
            var nextStep = this.getNextStep();
            var previousStep = this.getPreviousStep();

            // Use closures to keep a reference to the actual next and previous
            // steps for this form.
            function showNextStep() {
                _this.showStep(nextStep)
            }

            function showPreviousStep() {
                _this.showStep(previousStep);
            }

            function trigger(defaults) {
                _this.trigger(base.FormView.SHOW_FORM, form, showNextStep,
                    showPreviousStep, customButtons, defaults);
            }

            if (form === 'model-settings') {
                var locationFile = new models.LocationFile({
                    filename: this.locationFileMeta.get('filename')
                }, {
                    gnomeModel: this.gnomeModel
                });

                locationFile.once('sync', function(model) {
                    trigger(model.attributes);
                });

                locationFile.fetch();
            } else {
                trigger();
            }
        }
    }, {
        LOCATION_CHOSEN: 'locationFileWizardFormView:locationChosen'
    });

    return {
        LocationFileWizardFormView: LocationFileWizardFormView
    }
});