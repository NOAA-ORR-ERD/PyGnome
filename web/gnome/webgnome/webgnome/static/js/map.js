
// Get an alias to the `window.noaa.erd.gnome` namespace.
gnome = window.noaa.erd.gnome;

gnome.MapModel = function (opts) {
    this.bbox = opts.bbox;
    this.zoomLevel = opts.zoomLevel == undefined ? 4 : opts.zoomLevel;
    this.id = opts.id;

    if (this.id === undefined) {
        // TODO: Set an error state so the UI becomes unusable.
        alert('Failed to connect to a running model. Please reload.')
    }
};

// URLs
gnome.MapModel.RUN_URL = '/model/${id}/run';
gnome.MapModel.ZOOM_URL = '/model/${id}/zoom';

// Events
gnome.MapModel.MODEL_RELOADED = 'gnome:modelReloaded';

gnome.MapModel.prototype = {
    getRect: function(startPosition, endPosition) {
         var newStartPosition, newEndPosition;

        // Do a shallow object copy, so we don't modify the original.
        if (endPosition.x > startPosition.x || endPosition.y > startPosition.y) {
            newStartPosition = $.extend({}, startPosition);
            newEndPosition = $.extend({}, endPosition);
        } else {
            newStartPosition = $.extend({}, endPosition);
            newEndPosition = $.extend({}, startPosition);
        }       
        
        return {start: newStartPosition, end: newEndPosition};
    },
    
    // Adjust a selection rectangle so that it fits within the bounding box.
    getAdjustedRect: function(rect) {
        var adjustedRect = this.getRect(rect[0], rect[1]);

        if (adjustedRect.start.x > this.bbox[0].x) {
            adjustedRect.start.x = this.bbox[0].x;
        }
        if (adjustedRect.start.y < this.bbox[0].y) {
            adjustedRect.start.y = this.bbox[0].y;
        }       
        
        if (adjustedRect.end.x < this.bbox[1].x) {
            adjustedRect.end.x = this.bbox[1].x;
        }
        if (adjustedRect.end.y > this.bbox[1].y) {
            adjustedRect.end.y = this.bbox[1].y;
        }

        return adjustedRect;
    },
    
    areCoordinatesInsideMap: function(position) {
        return (position.x > this.bbox[0].x && position.x < this.bbox[1].x
            && position.y > this.bbox[0].y && position.y < this.bbox[1].y);
    },

    isRectInsideMap: function (rect) {
        var _rect = this.getRect(rect[0], rect[1]);

        return this.areCoordinatesInsideMap(_rect.start) &&
               this.areCoordinatesInsideMap(_rect.end);
    },

    zoom: function(rect) {
        var args = {
            id: this.id,
            zoomLevel: this.zoomLevel,
            rect: rect
        };

        console.log(args);
    }
};


gnome.MapView = function(opts) {
    this.imageEl = $(opts.imageEl);
};

gnome.MapView.DRAGGING_STOPPED = 'gnome:draggingStopped';

gnome.MapView.prototype = {
    initialize: function() {
        var _this = this;

        this.imageEl.selectable({
            start: function (event) {
                _this.startPosition = {x: event.pageX, y: event.pageY};
            },
            stop: function (event) {
                $(_this).trigger(
                    gnome.MapView.DRAGGING_STOPPED,
                    [_this.startPosition, {x: event.pageX, y: event.pageY}]);
            }
        });
    },

    getSize: function () {
        return {height:this.imageEl.height(), width:this.imageEl.width()}
    },

    getBoundingBox: function() {
        var pos = this.imageEl.position();
        var size = this.imageEl.size();

        return [
            {x: pos.left, y: pos.top},
            {x: pos.left + size.width, y: pos.top + size.height}
        ];
    }
};


gnome.MapController = function(mapEl) {
    var _this = this;

    this.mapView = new gnome.MapView({
        imageEl: mapEl
    });

    $(this.mapView).bind(gnome.MapView.DRAGGING_STOPPED, function(event, startPosition, endPosition) {
        _this.zoom([startPosition, endPosition]);
    });

    this.mapView.initialize();

    this.mapModel = new gnome.MapModel({
        // XXX: Does it make more sense to get this value from the server?
        bbox: this.mapView.getBoundingBox(),
        // TODO: Get ID from the server.
        id: 1
    });

     $(this.mapModel).bind(gnome.MapModel.MODEL_RELOADED, function(data) {
        _this.mapView.refresh(data);
    });

    return this;
};

gnome.MapController.prototype = {
    zoom: function (rect) {
        var isInsideMap = this.mapModel.isRectInsideMap(rect);

        if (!isInsideMap) {
            rect = this.mapModel.getAdjustedRect(rect);
        }

        this.mapModel.zoom(rect);
    }
};


$(window).ready(function() {
    new gnome.MapController('#map');

    // Attach the dynatree widget to an existing <div id="tree"> element
    // and pass the tree options as an argument to the dynatree() function:
    $("#tree").dynatree({
        onActivate: function(node) {
            // A DynaTreeNode object is passed to the activation handler
            // Note: we also get this event, if persistence is on, and the page is reloaded.
            alert("You activated " + node.data.title);
        },
        persist: true,
        children: [ // Pass an array of nodes.
            {title: "Model Settings",
                children: [
                    {title: "Sub-item 2.1"},
                    {title: "Sub-item 2.2"}
                ]
            },
            {title: "Universal Movers",
                children: [
                    {title: "Sub-item 2.1"},
                    {title: "Sub-item 2.2"}
                ]
            },
            {title: "Maps",
                children: [
                    {title: "Sub-item 2.1"},
                    {title: "Sub-item 2.2"}
                ]
            },
            {title: "Spills",
                children: [
                    {title: "Sub-item 2.1"},
                    {title: "Sub-item 2.2"}
                ]
            }
        ]
    });
});

