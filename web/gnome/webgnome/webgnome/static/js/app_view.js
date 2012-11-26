
define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'lib/mousetrap',
    'models',
    'views',
    'util'
], function($, _, Backbone, Mousetrap, models, views, util) {

     /*
     `AppView` acts as a controller, listening to delegate views and models for
     events and coordinating any necessary changes.

     As a design principle, `AppView` should only handle events triggered by models
     and views that *require* coordination. Otherwise, views should listen directly
     to a specific model (or another view) and handle updating themselves without
     assistance from `AppView`. This convention is in-progress and could be better
     enforced.
     */
    var AppView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);

            this.apiRoot = "/model";

            this.setupForms();

            // Initialize the model with any previously-generated time step data the
            // server had available.
            this.model = new models.Model(this.options.generatedTimeSteps, {
                url: this.apiRoot,
                expectedTimeSteps: this.options.expectedTimeSteps,
                currentTimeStep: this.options.currentTimeStep
            });

            this.menuView = new views.MenuView({
                // XXX: Hard-coded IDs
                modelDropdownEl: "#file-drop",
                runDropdownEl: "#run-drop",
                helpDropdownEl: "#help-drop",
                newItemEl: "#menu-new",
                runItemEl: "#menu-run",
                stepItemEl: "#menu-step",
                runUntilItemEl: "#menu-run-until"
            });

            this.sidebarEl = '#' + this.options.sidebarId;

            this.treeView = new views.TreeView({
                // XXX: Hard-coded URL, ID.
                treeEl: "#tree",
                url: "/tree",
                ajaxForms: this.forms,
                model: this.model
            });

            this.treeControlView = new views.TreeControlView({
                // XXX: Hard-coded IDs
                addButtonEl: "#add-button",
                removeButtonEl: "#remove-button",
                settingsButtonEl: "#settings-button",
                treeView: this.treeView
            });

            this.mapView = new views.MapView({
                mapEl: '#' + this.options.mapId,
                placeholderEl: '#' + this.options.mapPlaceholderId,
                backgroundImageUrl: this.options.backgroundImageUrl,
                frameClass: 'frame',
                activeFrameClass: 'active',
                model: this.model
            });

            this.mapControlView = new views.MapControlView({
                // XXX: Hard-coded IDs.
                sliderEl: "#slider",
                playButtonEl: "#play-button",
                pauseButtonEl: "#pause-button",
                backButtonEl: "#back-button",
                forwardButtonEl: "#forward-button",
                zoomInButtonEl: "#zoom-in-button",
                zoomOutButtonEl: "#zoom-out-button",
                moveButtonEl: "#move-button",
                fullscreenButtonEl: "#fullscreen-button",
                resizeButtonEl: "#resize-button",
                timeEl: "#time",
                // XXX: Partially hard-coded URL.
                url: this.apiRoot + '/time_steps',
                model: this.model,
                mapView: this.mapView
            });

            this.messageView = new views.MessageView({
                model: this.model,
                ajaxForms: this.forms
            });

            this.setupEventHandlers();
            this.setupKeyboardHandlers();

            // Setup datepickers
            _.each($('.date'), function(field) {
                $(field).datepicker({
                    changeMonth: true,
                    changeYear: true
                });
            });
        },

        setupEventHandlers: function() {
            this.model.on(models.Model.RUN_ERROR, this.modelRunError);
            this.treeView.on(views.TreeView.ITEM_DOUBLE_CLICKED, this.treeItemDoubleClicked);
            this.formViews.on(views.ModalFormViewContainer.REFRESHED, this.refreshForms);

            this.treeControlView.on(views.TreeControlView.ADD_BUTTON_CLICKED, this.addButtonClicked);
            this.treeControlView.on(views.TreeControlView.REMOVE_BUTTON_CLICKED, this.removeButtonClicked);
            this.treeControlView.on(views.TreeControlView.SETTINGS_BUTTON_CLICKED, this.settingsButtonClicked);

            this.mapControlView.on(views.MapControlView.PLAY_BUTTON_CLICKED, this.playButtonClicked);
            this.mapControlView.on(views.MapControlView.PAUSE_BUTTON_CLICKED, this.pause);
            this.mapControlView.on(views.MapControlView.ZOOM_IN_BUTTON_CLICKED, this.enableZoomIn);
            this.mapControlView.on(views.MapControlView.ZOOM_OUT_BUTTON_CLICKED, this.enableZoomOut);
            this.mapControlView.on(views.MapControlView.SLIDER_CHANGED, this.sliderChanged);
            this.mapControlView.on(views.MapControlView.SLIDER_MOVED, this.sliderMoved);
            this.mapControlView.on(views.MapControlView.BACK_BUTTON_CLICKED, this.jumpToFirstFrame);
            this.mapControlView.on(views.MapControlView.FORWARD_BUTTON_CLICKED, this.jumpToLastFrame);
            this.mapControlView.on(views.MapControlView.FULLSCREEN_BUTTON_CLICKED, this.useFullscreen);
            this.mapControlView.on(views.MapControlView.RESIZE_BUTTON_CLICKED, this.disableFullscreen);

            this.mapView.on(views.MapView.PLAYING_FINISHED, this.stopAnimation);
            this.mapView.on(views.MapView.DRAGGING_FINISHED, this.zoomIn);
            this.mapView.on(views.MapView.FRAME_CHANGED, this.frameChanged);
            this.mapView.on(views.MapView.MAP_WAS_CLICKED, this.zoomOut);

            this.menuView.on(views.MenuView.NEW_ITEM_CLICKED, this.newMenuItemClicked);
            this.menuView.on(views.MenuView.RUN_ITEM_CLICKED, this.runMenuItemClicked);
            this.menuView.on(views.MenuView.RUN_UNTIL_ITEM_CLICKED, this.runUntilMenuItemClicked);

            this.addMoverFormView.on(views.AddMoverFormView.MOVER_CHOSEN, this.moverChosen);
        },

        setupKeyboardHandlers: function() {
            var _this = this;

            Mousetrap.bind('space', function() {
                util.log('toggle playing')
                if (_this.mapControlView.isPlaying()) {
                    _this.pause();
                } else {
                    _this.play({});
                }
            });

            Mousetrap.bind('o', function() {
                util.log('open item');
                _this.showFormForActiveTreeItem();
            });

            Mousetrap.bind('n o', function() {
                util.log('new model');
                _this.newMenuItemClicked();
            });

            Mousetrap.bind('n m', function() {
                util.log('new mover');
                _this.showFormWithId('AddMoverForm');
            });

            Mousetrap.bind('n w', function() {
                util.log('new wind mover');
                _this.showFormWithId('WindMoverForm');
            });

            Mousetrap.bind('s f', function() {
                util.log('save form')
                var visibleSaveButton = $('.modal[aria-hidden=false] .btn-primary');
                if (visibleSaveButton) {
                    visibleSaveButton.click();
                }
            });
        },

        destroyForms: function() {
            if (this.forms) {
                this.forms.deleteAll();
            }

            if (this.formViews) {
                this.formViews.deleteAll();
            }
        },

        refreshForms: function() {
            this.destroyForms();
            this.addForms();
            util.fixModals();
        },

        addForms: function() {
            var _this = this;

            this.addMoverFormView = new views.AddMoverFormView({
                el: $('#' + this.options.addMoverFormId),
                formContainerEl: '#' + this.options.formContainerId
            });

            this.formViews.add(this.options.addMoverFormId, this.addMoverFormView);

            // Create an `AjaxForm` and bind it to a `ModalFormView` for each modal
            // form on the page, other than the Add Mover form, which we handled.
            _.each($('div.modal'), function(modalDiv) {
                var $div = $(modalDiv);
                var $form = $div.find('form');
                var modalFormId = $div.attr('id');

                if (modalFormId === _this.options.addMoverFormId) {
                    return;
                }

                _this.forms.add({
                    id: modalFormId,
                    url: $form.attr('action')
                });

                var ajaxForm = _this.forms.get(modalFormId);
                var formEl = $('#' + modalFormId);
                var formContainerEl = '#' + _this.options.formContainerId;

                if ($div.hasClass('wind')) {
                    _this.formViews.add(modalFormId, new views.WindMoverFormView({
                        ajaxForm: ajaxForm,
                        el: formEl,
                        formContainerEl: formContainerEl
                    }));
                } else {
                    _this.formViews.add({
                        id: modalFormId,
                        ajaxForm: ajaxForm,
                        el: formEl,
                        formContainerEl: formContainerEl
                    });
                }
            });
        },

        setupForms: function() {
            // `AjaxForm` instances, keyed to form ID.
            this.forms = new models.AjaxFormCollection();

            this.formViews = new views.ModalFormViewContainer({
                el: $('#' + this.options.formContainerId),
                ajaxForms: this.forms,
                url: this.options.formsUrl
            });

            this.addForms();
        },

        modelRunError: function() {
            this.messageView.displayMessage({
                type: 'error',
                text: 'Model run failed.'
            });
        },

        isValidFormType: function(formType) {
            return _.contains(this.formTypes, formType);
        },

        runMenuItemClicked: function() {
            this.play({});
        },

        runUntilMenuItemClicked: function() {
            this.fetchForm({type: 'run_until'});
        },

        newMenuItemClicked: function() {
            if (!confirm("Reset model?")) {
                return;
            }

            this.model.create();
        },

        play: function(opts) {
            this.mapControlView.disableControls();
            this.mapControlView.enableControls([this.mapControlView.pauseButtonEl]);
            this.mapControlView.setPlaying();
            this.mapView.setPlaying();

            if (this.model.isOnLastTimeStep()) {
                this.model.rewind();
            }

            this.model.run(opts);
        },

        playButtonClicked: function() {
            this.play({});
        },

        enableZoomIn: function() {
            if (this.model.hasData() === false) {
                return;
            }

            this.mapControlView.setZoomingIn();
            this.mapView.makeActiveImageClickable();
            this.mapView.makeActiveImageSelectable();
            this.mapView.setZoomingInCursor();
        },

        enableZoomOut: function() {
            if (this.model.hasData() === false) {
                return;
            }

            this.mapControlView.setZoomingOut();
            this.mapView.makeActiveImageClickable();
            this.mapView.setZoomingOutCursor();
        },

        stopAnimation: function() {
            this.mapControlView.setStopped();
        },

        zoomIn: function(startPosition, endPosition) {
            this.mapView.setPaused();
            this.mapControlView.setPaused();
            this.model.rewind();

            if (endPosition) {
                var rect = {start: startPosition, end: endPosition};
                var isInsideMap = this.mapView.isRectInsideMap(rect);

                // If we are at zoom level 0 and there is no map portion outside of
                // the visible area, then adjust the coordinates of the selected
                // rectangle to the on-screen pixel bounds.
                if (!isInsideMap && this.model.zoomLevel === 0) {
                    rect = this.mapView.getAdjustedRect(rect);
                }

                this.model.zoomFromRect(rect, Model.ZOOM_IN);
            } else {
                this.model.zoomFromPoint(startPosition, Model.ZOOM_IN);
            }

            this.mapView.setRegularCursor();
        },

        zoomOut: function(point) {
            this.model.rewind();
            this.mapView.setPaused();
            this.mapControlView.setPaused();
            this.model.zoomFromPoint(point, Model.ZOOM_OUT);
            this.mapView.setRegularCursor();
        },

        pause: function() {
            this.mapView.setPaused();
            this.mapControlView.setPaused();
            this.mapControlView.enableControls();
        },

        sliderChanged: function(newStepNum) {
            // No need to do anything if the slider is on the current time step.
            if (newStepNum === this.model.currentTimeStep) {
                return;
            }

            // If the model and map view have the time step, display it.
            if (this.model.hasCachedTimeStep(newStepNum) &&
                    this.mapView.timeStepIsLoaded(newStepNum)) {
                this.model.setCurrentTimeStep(newStepNum);
                return;
            }

            // Otherwise, we need to run until the new time step.
            this.play({
                runUntilTimeStep: newStepNum
            });
        },

        frameChanged: function() {
            if (this.mapView.isPaused() || this.mapView.isStopped()) {
                return;
            }
            this.model.getNextTimeStep();
        },

        jumpToFirstFrame: function() {
            this.model.setCurrentTimeStep(0);
        },

        /*
         Jump to the last LOADED frame of the animation. This will stop at
         whatever frame was the last received from the server.

         TODO: This should probably do something fancier, like block and load
         all of the remaining frames if they don't exist, until the end.
         */
        jumpToLastFrame: function() {
            var lastFrame = this.model.length - 1;
            this.model.setCurrentTimeStep(lastFrame);
        },

        useFullscreen: function() {
            this.mapControlView.switchToFullscreen();
            $(this.sidebarEl).hide('slow');
        },

        disableFullscreen: function() {
            this.mapControlView.switchToNormalScreen();
            $(this.sidebarEl).show('slow');
        },

        showFormWithId: function(formId) {
            var formView = this.formViews.get(formId);

            if (formView === undefined) {
                return;
            }

            formView.show();
        },

        showFormForNode: function(node) {
            var formView = this.formViews.get(node.data.form_id);

            if (formView === undefined) {
                return;
            }

            if (node.data.id) {
                formView.reload(node.data.id);
            } else {
                formView.show();
            }
        },

        /*
         Show the `ModalFormView` for the active tree item.

         If showing an add form, display the `ModalFormView` for the active node.

         If showing an edit form, perform a `fetch` using the `AjaxForm` for the
         selected node first, which will trigger the bound `ModalFormView` to display.

         The distinction of "add" versus "edit" is made on whether or not the node
         has an `id` property with a non-null value.
         */
        showFormForActiveTreeItem: function() {
            var node = this.treeView.getActiveItem();
            this.showFormForNode(node);
        },

        addButtonClicked: function() {
            this.showFormForActiveTreeItem();
        },

        treeItemDoubleClicked: function(node) {
            this.showFormForNode(node);
        },

        settingsButtonClicked: function() {
            this.showFormForActiveTreeItem();
        },

        removeButtonClicked: function() {
            var node = this.treeView.getActiveItem();

            if (!node.data.form_id || !node.data.id) {
                return;
            }

            var type = node.data.form_id.replace('_', ' ');

            if (confirm('Remove ' + type + '?') === false) {
                return;
            }

            this.ajaxForm.submit({
                url: this.ajaxForm.get('url') + '/' + node.data.form_id + '/delete',
                data: "mover_id=" + node.data.id,
                error: function() {
                    alert('Could not remove item.');
                }
            });
        },

        moverChosen: function(moverType) {
            var formView = this.formViews.get(moverType);

            if (formView === undefined) {
                return;
            }

            formView.show();
        }
    });

    return {
        AppView: AppView
    };

});

