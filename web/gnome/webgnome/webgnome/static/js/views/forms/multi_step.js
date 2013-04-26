define([
    'jquery',
    'lib/underscore',
    'views/forms/modal',
    'views/forms/base',
], function($, _, modal, base) {
    var MultiStepFormView = modal.JQueryUIModalFormView.extend({
        events: {
            'click .ui-button.next': 'next',
            'click .ui-button.back': 'back'
        },

        initialize: function() {
            // Extend prototype's events with ours.
            this.events = _.extend({}, base.FormView.prototype.events, this.events);

            // Have to initialize super super before showing current step.
            MultiStepFormView.__super__.initialize.apply(this, arguments);

            this.annotateSteps();
        },

        /*
         Annotate each step div with its step number. E.g., the first step will
         have step number 0, second step number 1, etc.
         */
        annotateSteps: function() {
            _.each(this.$el.find('.step'), function(step, idx) {
                $(step).attr('data-step', idx);
            });
        },

        getCurrentStep: function() {
            return this.$el.find('.step').not('.hidden');
        },

        getCurrentStepNum: function() {
            return this.getCurrentStep().data('step');
        },

        getStep: function(stepNum) {
            return this.$el.find('.step[data-step="' + stepNum + '"]');
        },

        getNextStep: function() {
            var currentStep = this.getCurrentStep();
            var nextStepNum = currentStep.data('step') + 1;
            return this.getStep(nextStepNum);
        },

        getPreviousStep: function() {
            var currentStep = this.getCurrentStep();
            var previousStepNum = currentStep.data('step') - 1;
            // Minimum step number is 0.
            previousStepNum = previousStepNum < 0 ? 0 : previousStepNum;
            return this.getStep(previousStepNum);
        },

        show: function() {
            this.showStep(this.getStep(0));
            MultiStepFormView.__super__.show.apply(this, arguments);
        },

        showStep: function(step) {
            if (step.length) {
                this.$el.find('.step').not(step).addClass('hidden');
                step.removeClass('hidden');
            }
        },

        next: function() {
            var nextStep = this.getNextStep();
            this.showStep(nextStep);
        },

        back: function() {
            var previousStep = this.getPreviousStep();
            this.showStep(previousStep);
        }
    });

    return {
        MultiStepFormView: MultiStepFormView
    }
});