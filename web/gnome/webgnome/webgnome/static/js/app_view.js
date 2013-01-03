
define([
    'jquery',
    'lib/underscore',
    'lib/backbone',
    'lib/mousetrap',
    'models',
    'forms',
    'views',
    'util'
], function($, _, Backbone, Mousetrap, models, forms, views, util) {

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

            this.apiRoot = "/model/" + this.options.modelId;

            // Initialize the model with any previously-generated time step data the
            // server had available.
            this.model = new models.ModelRun(this.options.generatedTimeSteps, {
                url: this.apiRoot,
                expectedTimeSteps: this.options.expectedTimeSteps,
                currentTimeStep: this.options.currentTimeStep,
                bounds: this.options.mapBounds || []
            });

            this.setupModels();
            this.setupForms();

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
                url: this.apiRoot + "/tree",
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
                spillButtonEl: "#spill-button",
                timeEl: "#time",
                // XXX: Partially hard-coded URL.
                url: this.apiRoot + '/time_steps',
                model: this.model,
                mapView: this.mapView
            });

            this.messageView = new views.MessageView({
                model: this.model
                // ajaxForms
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
            this.model.on(models.ModelRun.CREATED, this.newModelCreated);
            this.model.on(models.ModelRun.RUN_ERROR, this.modelRunError);
            this.pointReleaseSpills.on("update", this.spillUpdated);

            this.formViews.on(forms.FormViewContainer.REFRESHED, this.mapView.drawSpills);
            this.formViews.on(forms.FormView.REFRESHED, this.mapView.drawSpills);
            this.addSpillFormView.on(forms.AddSpillFormView.CANCELED, this.mapView.drawSpills);
            this.formViews.on(forms.FormViewContainer.REFRESHED, this.refreshForms);

            this.treeView.on(views.TreeView.ITEM_DOUBLE_CLICKED, this.treeItemDoubleClicked);

            this.treeControlView.on(views.TreeControlView.ADD_BUTTON_CLICKED, this.addButtonClicked);
            this.treeControlView.on(views.TreeControlView.REMOVE_BUTTON_CLICKED, this.removeButtonClicked);
            this.treeControlView.on(views.TreeControlView.SETTINGS_BUTTON_CLICKED, this.settingsButtonClicked);

            this.mapControlView.on(views.MapControlView.PLAY_BUTTON_CLICKED, this.playButtonClicked);
            this.mapControlView.on(views.MapControlView.PAUSE_BUTTON_CLICKED, this.pause);
            this.mapControlView.on(views.MapControlView.ZOOM_IN_BUTTON_CLICKED, this.enableZoomIn);
            this.mapControlView.on(views.MapControlView.ZOOM_OUT_BUTTON_CLICKED, this.enableZoomOut);
            this.mapControlView.on(views.MapControlView.SLIDER_CHANGED, this.sliderChanged);
            this.mapControlView.on(views.MapControlView.BACK_BUTTON_CLICKED, this.rewind);
            this.mapControlView.on(views.MapControlView.FORWARD_BUTTON_CLICKED, this.jumpToLastFrame);
            this.mapControlView.on(views.MapControlView.FULLSCREEN_BUTTON_CLICKED, this.useFullscreen);
            this.mapControlView.on(views.MapControlView.RESIZE_BUTTON_CLICKED, this.disableFullscreen);
            this.mapControlView.on(views.MapControlView.SPILL_BUTTON_CLICKED, this.enableSpillDrawing);

            this.mapView.on(views.MapView.PLAYING_FINISHED, this.stopAnimation);
            this.mapView.on(views.MapView.DRAGGING_FINISHED, this.zoomIn);
            this.mapView.on(views.MapView.FRAME_CHANGED, this.frameChanged);
            this.mapView.on(views.MapView.MAP_WAS_CLICKED, this.zoomOut);
            this.mapView.on(views.MapView.SPILL_DRAWN, this.spillDrawn);

            this.menuView.on(views.MenuView.NEW_ITEM_CLICKED, this.newMenuItemClicked);
            this.menuView.on(views.MenuView.RUN_ITEM_CLICKED, this.runMenuItemClicked);
            this.menuView.on(views.MenuView.RUN_UNTIL_ITEM_CLICKED, this.runUntilMenuItemClicked);
        },

        /*
         Consider the model dirty if the user updates an existing spill, so
         we don't get cached images back on the next model run.
         */
        spillUpdated: function(form) {
            this.rewind();
        },

        setupKeyboardHandlers: function() {
            var _this = this;

            Mousetrap.bind('space', function() {
                if (_this.mapControlView.isPlaying()) {
                    _this.pause();
                } else {
                    _this.play({});
                }
            });

            Mousetrap.bind('o', function() {
                _this.showFormForActiveTreeItem();
            });

            Mousetrap.bind('n o', function() {
                _this.newMenuItemClicked();
            });

            Mousetrap.bind('n m', function() {
                _this.formViews.hideAll();
                _this.showFormWithId('add_mover');
            });

            Mousetrap.bind('n w', function() {
                _this.formViews.hideAll();
                _this.showFormWithId('wind_mover');
            });

            Mousetrap.bind('s f', function() {
                var visibleSaveButton = $('div.form[hidden=false] .btn-primary');
                if (visibleSaveButton) {
                    visibleSaveButton.click();
                }
            });
        },

        spillDrawn: function(x, y) {
            this.addSpillFormView.show([x, y]);
        },

        newModelCreated: function() {
            this.formViews.refresh();
        },

        setupForms: function() {
            this.formViews = new forms.FormViewContainer({
                el: $('#' + this.options.formContainerId),
                pointReleaseSpills: this.pointReleaseSpills,
                windMovers: this.windMovers,
                model: this.model // TODO: Remove this when we remove Long Island
            });

            this.addForms();
        },

        destroyForms: function() {
            if (this.formViews) {
                this.formViews.deleteAll();
            }
        },

        refreshForms: function() {
            this.destroyForms();
            this.addForms();
        },

        addForms: function() {
            this.addMoverFormView = new forms.AddMoverFormView({
                el: $('#' + this.options.addMoverFormId),
                formContainerEl: '#' + this.options.formContainerId
            });

            this.addSpillFormView = new forms.AddSpillFormView({
                el: $('#' + this.options.addSpillFormId),
                formContainerEl: '#' + this.options.formContainerId
            });

            this.modelSettingsFormView = new forms.ModalFormView({
                el: $('#' + this.options.modelSettingsFormId),
                formContainerEl: '#' + this.options.formContainerId,
                model: this.modelSettings
            });

            this.addWindMoverFormView = new forms.AddWindMoverFormView({
                // TODO: Hard-coded ID here
                el: $('#wind_mover')
            });

            this.addPointReleaseSpillFormView = new forms.AddPointReleaseSpillFormView({
                // TODO: Hard-coded ID here
                el: $('#point_release_spill')
            });

            this.addMoverFormView.on(forms.AddMoverFormView.MOVER_CHOSEN, this.moverChosen);
            this.addSpillFormView.on(forms.AddSpillFormView.SPILL_CHOSEN, this.spillChosen);

            this.formViews.add(this.options.addMoverFormId, this.addMoverFormView);
            this.formViews.add(this.options.addSpillFormId, this.addSpillFormView);
            this.formViews.add(this.options.modelSettingsFormId, this.modelSettingsFormView);
            this.formViews.add(this.options.windMoverFormId, this.addWindMoverFormView);
            this.formViews.add(this.options.pointReleaseSpillFormId, this.addPointReleaseSpillFormView);
        },

        setupModels: function() {
            this.pointReleaseSpills = new models.PointReleaseSpillCollection(
                this.options.pointReleaseSpills, {
                    url: "/model/" + this.options.modelId + "/spill/point_release"
                }
            );

            this.windMovers = new models.WindMoverCollection(
                this.options.windMovers, {
                    url: "/model/" + this.options.modelId + "/mover/wind"
                }
            );

            this.modelSettings = new models.ModelSettings(
                this.options.modelSettings, {
                    url: "/model/" + this.options.modelId +
                        "?include_movers=false&include_spills=false"
                }
            );
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
            // TODO: Fix this - old code.
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

                this.model.zoomFromRect(rect, models.ModelRun.ZOOM_IN);
            } else {
                this.model.zoomFromPoint(startPosition, models.ModelRun.ZOOM_IN);
            }

            this.mapView.setRegularCursor();
        },

        zoomOut: function(point) {
            this.model.rewind();
            this.mapView.setPaused();
            this.mapControlView.setPaused();
            this.model.zoomFromPoint(point, models.ModelRun.ZOOM_OUT);
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

        reset: function() {
            this.mapView.reset();
            this.model.clearData();
            this.mapControlView.reset();
        },

        rewind: function() {
            this.mapView.clear();
            this.model.clearData();
            this.mapControlView.reset();
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

        enableSpillDrawing: function() {
            this.mapView.canDrawSpill = true;
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
                formView.reload({id: node.data.id});
            } else {
                this.formViews.hideAll();
                formView.show();
            }
        },

        /*
         Show the `AjaxFormView` for the active tree item.

         If showing an add form, display the `AjaxFormView` for the active node.

         If showing an edit form, perform a `fetch` using the `AjaxForm` for the
         selected node first, which will trigger the bound `AjaxFormView` to display.

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

            function error() {
                alert('Error! Could not delete ' + node.data.title + '.');
            }

            if (!node.data.form_id || !node.data.object_id) {
                return error();
            }

            var ajaxForm = this.forms.get(node.data.delete_form_id);
            var formView = this.formViews.get(node.data.delete_form_id);

            if (!ajaxForm || !formView) {
                return error();
            }

            if (confirm('Remove ' + node.data.title + '?') === false) {
                return;
            }

            ajaxForm.submit({
                data: "obj_id=" + node.data.object_id,
                error: error
            });
        },

        moverChosen: function(moverType) {
            var formView = this.formViews.get(moverType);

            if (formView === undefined) {
                return;
            }

            formView.show();
        },

        spillChosen: function(spillType, coords) {
            var formView = this.formViews.get(spillType);

            if (formView === undefined) {
                return;
            }

            formView.show(coords);
        }
    });

    return {
        AppView: AppView
    };

});

