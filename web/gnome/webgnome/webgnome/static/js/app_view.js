
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
     events and coordinating state changes.

     `AppView` should only handle events triggered by models and views that
     *require* coordination. When possible, views should listen directly to a
     specific model (or another view) and handle updating themselves without
     assistance from `AppView`.
     */
    var AppView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this);

            // Used to construct model and other server-side URLs.
            this.apiRoot = "/model/" + this.options.modelId;
            this.router = this.options.router;

            this.setupModels();
            this.setupForms();

            this.menuView = new views.MenuView({
                modelDropdownEl: "#file-drop",
                runDropdownEl: "#run-drop",
                helpDropdownEl: "#help-drop",
                newItemEl: "#menu-new",
                runItemEl: "#menu-run",
                stepItemEl: "#menu-step",
                runUntilItemEl: "#menu-run-until"
            });

            this.sidebarEl = '#sidebar';
            $(this.sidebarEl).resizable({
                handles: 'e, w',
                resize: function (event, ui) {
                    $(this).css("height", '100%');
                }
            });

            this.treeView = new views.TreeView({
                treeEl: "#tree",
                apiRoot: this.apiRoot,
                gnomeRun: this.gnomeRun,
                gnomeSettings: this.gnomeSettings,
                map: this.map,
                customMap: this.customMap,
                collections: [
                    this.windMovers, this.randomMovers, this.surfaceReleaseSpills
                ]
            });

            this.treeControlView = new views.TreeControlView({
                addButtonEl: "#add-button",
                removeButtonEl: "#remove-button",
                settingsButtonEl: "#settings-button",
                treeView: this.treeView
            });

            this.mapView = new views.MapView({
                mapEl: '#map',
                placeholderClass: 'placeholder',
                frameClass: 'frame',
                activeFrameClass: 'active',
                gnomeRun: this.gnomeRun,
                model: this.map,
                animationThreshold: this.options.animationThreshold,
                newModel: this.options.newModel
            });

            this.mapControlView = new views.MapControlView({
                sliderEl: "#slider",
                sliderShadedEl: '#slider-shaded',
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
                gnomeRun: this.gnomeRun,
                mapView: this.mapView,
                model: this.map
            });

            this.messageView = new views.MessageView({
                gnomeRun: this.gnomeRun,
                gnomeSettings: this.gnomeSettings,
                surfaceReleaseSpills: this.surfaceReleaseSpills,
                windMovers: this.windMovers
            });

            this.splashView = new views.SplashView({
                el: $('#splash-page'),
                router: this.router
            });

            this.locationFileMapView = new views.LocationFileMapView({
                el: $('#location-file-map'),
                apiRoot: this.apiRoot,
                mapCanvas: '#map_canvas',
                locationFiles: this.options.locationFiles
            });

            this.setupEventHandlers();
            this.setupKeyboardHandlers();
        },

        setupEventHandlers: function() {
            var _this = this;
            this.customMap.on('sync', function() {
                _this.map.fetch();
            });

            this.gnomeRun.on(models.GnomeRun.RUN_ERROR, this.gnomeRunError);
            this.gnomeRun.on(models.GnomeRun.SERVER_RESET, this.rewind);

            this.surfaceReleaseSpills.on("sync", this.spillUpdated);
            this.surfaceReleaseSpills.on('sync', this.drawSpills);
            this.surfaceReleaseSpills.on('add', this.drawSpills);
            this.surfaceReleaseSpills.on('remove', this.drawSpills);

            this.addSpillFormView.on(forms.AddSpillFormView.CANCELED, this.drawSpills);
            this.addSurfaceReleaseSpillFormView.on(forms.SurfaceReleaseSpillFormView.CANCELED, this.drawSpills);
            this.editSurfaceReleaseSpillFormView.on(forms.SurfaceReleaseSpillFormView.CANCELED, this.drawSpills);

            this.formViews.on(forms.FormView.MESSAGE_READY, this.displayMessage);

            this.treeView.on(views.TreeView.ITEM_DOUBLE_CLICKED, this.treeItemDoubleClicked);

            this.treeControlView.on(views.TreeControlView.ADD_BUTTON_CLICKED, this.addButtonClicked);
            this.treeControlView.on(views.TreeControlView.REMOVE_BUTTON_CLICKED, this.removeButtonClicked);
            this.treeControlView.on(views.TreeControlView.SETTINGS_BUTTON_CLICKED, this.settingsButtonClicked);

            this.mapControlView.on(views.MapControlView.PLAY_BUTTON_CLICKED, this.playButtonClicked);
            this.mapControlView.on(views.MapControlView.PAUSE_BUTTON_CLICKED, this.pause);
            this.mapControlView.on(views.MapControlView.ZOOM_IN_BUTTON_CLICKED, this.enableZoomIn);
            this.mapControlView.on(views.MapControlView.ZOOM_OUT_BUTTON_CLICKED, this.enableZoomOut);
            this.mapControlView.on(views.MapControlView.SLIDER_MOVED, this.sliderMoved);
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
            this.mapView.on(views.MapView.READY, this.drawSpills);

            this.locationFileMapView.on(views.LocationFileMapView.LOCATION_CHOSEN, this.loadLocationFileWizard);

            this.menuView.on(views.MenuView.NEW_ITEM_CLICKED, this.newMenuItemClicked);
            this.menuView.on(views.MenuView.RUN_ITEM_CLICKED, this.runMenuItemClicked);
            this.menuView.on(views.MenuView.RUN_UNTIL_ITEM_CLICKED, this.runUntilMenuItemClicked);
            this.menuView.on(views.MenuView.LOCATION_FILE_ITEM_CLICKED, this.loadLocationFileWizard);
        },

        showForm: function(formId, success, cancel) {
            var formView = this.formViews.get(formId);
            if (formView) {
                formView.on(forms.FormView.SUBMITTED, success);
                formView.on(forms.FormView.CANCELED, cancel);
                formView.reload();
                formView.show();
            }
        },

        loadLocationFileWizardSuccess: function(data) {
            var html = $(data.html);
            html.appendTo($('#modal-container'));

            var formView = new forms.LocationFileWizardFormView({
                id: html.attr('id')
            });

            formView.on(forms.LocationFileWizardFormView.SHOW_FORM, this.showForm);
            formView.show();
        },

        loadLocationFileWizard: function(location) {
            models.getLocationFileWizard(
                this.apiRoot, location, {
                    success: this.loadLocationFileWizardSuccess
            });
        },

        loadLocationFile: function(location) {
            models.setLocationFile(this.apiRoot, location, {
                success: function() {
                    window.location = window.location.origin;
                }
            });
        },

        /*
         Consider the model dirty if the user updates an existing spill, so
         we don't get cached images back on the next model run.
         */
        spillUpdated: function() {
            this.rewind();
        },

        drawSpills: function() {
            this.mapView.drawSpills(this.surfaceReleaseSpills);
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
                _this.showFormWithId('add-wind-mover');
            });

            Mousetrap.bind('n p', function() {
                _this.formViews.hideAll();
                _this.showFormWithId('add-surface-release-spill');
            });

            Mousetrap.bind('s f', function() {
                var visibleSaveButton = $('div.form[hidden=false] .btn-primary');
                if (visibleSaveButton) {
                    visibleSaveButton.click();
                }
            });
        },

        spillDrawn: function(startCoords, endCoords) {
            this.addSpillFormView.show(startCoords, endCoords);
        },

        setupForms: function() {
            this.formViews = new forms.FormViewContainer({
                id: 'modal-container'
            });

            this.addMoverFormView = new forms.AddMoverFormView({
                id: 'add-mover'
            });

            this.addSpillFormView = new forms.AddSpillFormView({
                id: 'add-spill'
            });

            this.addMapFormView = new forms.AddMapFormView({
                id: 'add-map'
            });

            this.editMapFormView = new forms.MapFormView({
                id: 'edit-map',
                model: this.map,
                defaults: this.options.defaultMap
            });

            this.addCustomMapFormView = new forms.AddCustomMapFormView({
                id: 'add-custom-map',
                model: this.customMap,
                defaults: this.options.defaultCustomMap
            });

            this.addMapFromUploadFormView = new forms.AddMapFromUploadFormView({
                id: 'add-map-from-upload',
                model: this.map,
                uploadUrl: this.apiRoot + '/file_upload'
            });

            this.gnomeSettingsFormView = new forms.GnomeSettingsFormView({
                id: 'model-settings',
                model: this.gnomeSettings
            });

            this.addWindMoverFormView = new forms.AddWindMoverFormView({
                id: 'add-wind-mover',
                collection: this.windMovers,
                defaults: this.options.defaultWindMover,
                defaultWindTimeseriesValue: this.options.defaultWindTimeseriesValue
            });

            this.editWindMoverFormView = new forms.WindMoverFormView({
                id: 'edit-wind-mover',
                collection: this.windMovers,
                defaults: this.options.defaultWindMover,
                defaultWindTimeseriesValue: this.options.defaultWindTimeseriesValue
            });

            this.addRandomMoverFormView = new forms.AddRandomMoverFormView({
                id: 'add-random-mover',
                collection: this.randomMovers,
                defaults: this.options.defaultRandomMover,
            });

            this.editRandomMoverFormView = new forms.RandomMoverFormView({
                id: 'edit-random-mover',
                collection: this.randomMovers,
                defaults: this.options.defaultRandomMover,
            });

            this.addSurfaceReleaseSpillFormView = new forms.AddSurfaceReleaseSpillFormView({
                id: 'add-surface-release-spill',
                collection: this.surfaceReleaseSpills,
                defaults: this.options.defaultSurfaceReleaseSpill,
            });

            this.editSurfaceReleaseSpillFormView = new forms.SurfaceReleaseSpillFormView({
                id: 'edit-surface-release-spill',
                collection: this.surfaceReleaseSpills,
                defaults: this.options.defaultSurfaceReleaseSpill
            });

            this.addMoverFormView.on(forms.AddMoverFormView.MOVER_CHOSEN, this.moverChosen);
            this.addSpillFormView.on(forms.AddSpillFormView.SPILL_CHOSEN, this.spillChosen);
            this.addMapFormView.on(forms.AddMapFormView.SOURCE_CHOSEN, this.mapSourceChosen);

            this.formViews.add(this.addMoverFormView);
            this.formViews.add(this.addSpillFormView);
            this.formViews.add(this.addMapFormView);
            this.formViews.add(this.addMapFromUploadFormView);
            this.formViews.add(this.addWindMoverFormView);
            this.formViews.add(this.addRandomMoverFormView);
            this.formViews.add(this.addSurfaceReleaseSpillFormView);
            this.formViews.add(this.gnomeSettingsFormView);
            this.formViews.add(this.editWindMoverFormView);
            this.formViews.add(this.editRandomMoverFormView);
            this.formViews.add(this.editSurfaceReleaseSpillFormView);
            this.formViews.add(this.editMapFormView);
            this.formViews.add(this.addCustomMapFormView);
        },

        setupModels: function() {
            this.map = new models.Map(this.options.map, {
                url: this.apiRoot + '/map'
            });

            this.customMap = new models.CustomMap({}, {
                url: this.apiRoot + '/custom_map'
            });

            this.surfaceReleaseSpills = new models.SurfaceReleaseSpillCollection(
                this.options.surfaceReleaseSpills, {
                    url: this.apiRoot + "/spill/surface_release"
                }
            );

            this.windMovers = new models.WindMoverCollection(
                this.options.windMovers, {
                    url: this.apiRoot + "/mover/wind"
                }
            );

            this.randomMovers = new models.RandomMoverCollection(
                this.options.randomMovers, {
                    url: this.apiRoot + "/mover/random"
                }
            );

            this.gnomeSettings = new models.Gnome(this.options.gnomeSettings);

            // Initialize the model with any previously-generated time step data the
            // server had available.
            this.gnomeRun = new models.GnomeRun(this.options.generatedTimeSteps, {
                url: this.apiRoot,
                expectedTimeSteps: this.options.expectedTimeSteps,
                currentTimeStep: this.options.currentTimeStep,
                bounds: this.options.mapBounds || [],
                gnomeSettings: this.gnomeSettings
            });
        },

        displayMessage: function(message) {
            this.messageView.displayMessage(message);
        },

        gnomeRunError: function() {
            this.messageView.displayMessage({
                type: 'error',
                text: 'Model run failed.'
            });
        },

        runMenuItemClicked: function() {
            this.play({});
        },

        runUntilMenuItemClicked: function() {
            // TODO: Implement.
            console.log('run until item clicked');
        },

        newMenuItemClicked: function() {
            if (!confirm("Reset model?")) {
                return;
            }

            this.gnomeSettings.wasDeleted = true;
            this.gnomeSettings.destroy({
                success: function() {
                    util.Cookies.setItem('model_deleted', true);
                    window.location = window.location.origin;
                }
            });
        },

        play: function(opts) {
            if (!this.map.id) {
                window.alert('You must add a map before you can run the model.');
                return;
            }

            this.mapControlView.disableControls();
            this.mapControlView.enableControls([this.mapControlView.pauseButtonEl]);
            this.mapControlView.setPlaying();
            this.mapView.setPlaying();

            if (this.gnomeRun.isOnLastTimeStep()) {
                this.gnomeRun.rewind();
            }

            this.gnomeRun.run(opts);
        },

        playButtonClicked: function() {
            this.play({});
        },

        enableZoomIn: function() {
            if (this.gnomeRun.hasData() === false) {
                return;
            }

            this.mapControlView.setZoomingIn();
            this.mapView.makeActiveImageClickable();
            this.mapView.makeActiveImageSelectable();
            this.mapView.setZoomingInCursor();
        },

        enableZoomOut: function() {
            if (this.gnomeRun.hasData() === false) {
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
            this.gnomeRun.rewind();

            if (endPosition) {
                var rect = {start: startPosition, end: endPosition};
                var isInsideMap = this.mapView.isRectInsideMap(rect);

                // If we are at zoom level 0 and there is no map portion outside of
                // the visible area, then adjust the coordinates of the selected
                // rectangle to the on-screen pixel bounds.
                if (!isInsideMap && this.gnomeRun.zoomLevel === 0) {
                    rect = this.mapView.getAdjustedRect(rect);
                }

                this.gnomeRun.zoomFromRect(rect, models.GnomeRun.ZOOM_IN);
            } else {
                this.gnomeRun.zoomFromPoint(startPosition, models.GnomeRun.ZOOM_IN);
            }

            this.mapView.setRegularCursor();
        },

        zoomOut: function(point) {
            this.gnomeRun.rewind();
            this.mapView.setPaused();
            this.mapControlView.setPaused();
            this.gnomeRun.zoomFromPoint(point, models.GnomeRun.ZOOM_OUT);
            this.mapView.setRegularCursor();
        },

        pause: function() {
            this.mapView.setPaused();
            this.mapControlView.setPaused();
            this.mapControlView.enableControls();
        },

        sliderMoved: function(newStepNum) {
            // No need to do anything if the slider is on the current time step.
            if (newStepNum === this.gnomeRun.currentTimeStep) {
                return;
            }

            // If the model and map view have the time step, display it.
            if (this.gnomeRun.hasCachedTimeStep(newStepNum) &&
                    this.mapView.timeStepIsLoaded(newStepNum)) {
                this.gnomeRun.setCurrentTimeStep(newStepNum);
            }
        },

        sliderChanged: function(newStepNum) {
            // No need to do anything if the slider is on the current time step.
            if (newStepNum === this.gnomeRun.currentTimeStep) {
                return;
            }

            // If the model and map view don't have the time step,
            // we need to run until the new time step.
            if (!this.gnomeRun.hasCachedTimeStep(newStepNum)
                    || !this.mapView.timeStepIsLoaded(newStepNum)) {
                this.play({
                    runUntilTimeStep: newStepNum
                });
            }
        },

        frameChanged: function() {
            if (this.mapView.isPaused() || this.mapView.isStopped()) {
                return;
            }
            this.gnomeRun.getNextTimeStep();
        },

        reset: function() {
            this.mapView.reset();
            this.gnomeRun.clearData();
            this.mapControlView.reset();
        },

        rewind: function() {
            this.mapView.clear();
            this.gnomeRun.clearData();
            this.mapControlView.reset();

            if (this.map.id) {
                this.mapControlView.enableControls(
                    this.mapControlView.mapControls);
            }
        },

        /*
         Jump to the last LOADED frame of the animation. This will stop at
         whatever frame was the last received from the server.
         */
        jumpToLastFrame: function() {
            var lastFrame = this.gnomeRun.length - 1;
            this.gnomeRun.setCurrentTimeStep(lastFrame);
        },

        useFullscreen: function() {
            this.mapControlView.switchToFullscreen();
            $(this.sidebarEl).hide('slow');
        },

        disableFullscreen: function() {
            this.mapControlView.switchToNormalScreen();
            $(this.sidebarEl).removeClass('hidden');
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
            var formView;

            if (node.data.form_id) {
                formView = this.formViews.get(node.data.form_id);
            }

            if (formView === undefined) {
                return;
            }

            // This has to come before we show the form because form views
            // may set their models to null when hiding.
            this.formViews.hideAll();

            if (node.data.object_id) {
                try {
                    formView.reload(node.data.object_id);
                } catch (e) {
                    if (e instanceof forms.ModelNotFoundException) {
                        window.alert('That item is unavailable right now. ' +
                            'Please refresh the page and try again.');

                        return;
                    }
                }
            }

            formView.show();
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

        deleteObjectForNode: function(object, node) {
            if (confirm('Remove ' + node.data.title + '?') === false) {
                return;
            }

            object.destroy();
        },

        removeButtonClicked: function() {
            var node = this.treeView.getActiveItem();

            function error() {
                return alert('That item cannot be removed.')
            }

            if (!node.data.object_id || !node.data.form_id) {
                return error();
            }

            var formView = this.formViews.get(node.data.form_id);
            var model = formView.getModel(node.data.object_id);

            if (!formView || !model) {
                return error();
            }

            this.deleteObjectForNode(model, node);
        },

        moverChosen: function(moverType) {
            var formView = this.formViews.get(moverType);

            if (formView === undefined) {
                return;
            }

            formView.reload();
            formView.show();
        },

        spillChosen: function(spillType, startCoords, endCoords) {
            var formView = this.formViews.get(spillType);

            if (formView === undefined) {
                return;
            }

            formView.reload();
            formView.show(startCoords, endCoords);
        },

        mapSourceChosen: function(source) {
            var formView = this.formViews.get(source);

            if (formView === undefined) {
                return;
            }

            formView.show();
        },

        showSection: function(section) {
            var sectionViews = {
                'splash-page': this.splashView,
                'model': this.mapView,
                'location-file-map': this.locationFileMapView
            };

            if (section in sectionViews) {
                var sectionSel = '#' + section;
                var view = sectionViews[section];

                $('.section').not(sectionSel).addClass('hidden');
                $(sectionSel).removeClass('hidden');
                if (view.show) {
                    view.show();
                }
            }
        }
    });

    return {
        AppView: AppView
    };

});

