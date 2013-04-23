define([
    'models',
    'lib/moment',
    'lib/sinon'
], function(models) {

    module('TimeStep');

    test('get should convert timestamp fields to MM/DD/YYY HH:mm format', function() {
        var time = moment().toString();
        var step = new models.TimeStep({timestamp: time});
        ok(step.get('timestamp') === moment(time).format('MM/DD/YYYY HH:mm'));
    });


    module('BaseModel');

    test('collection is resorted if ID field changes', function() {
        var model = new models.BaseModel({id: 1});
        model.collection = {};
        model.collection.sort = sinon.spy();
        model.collection.comparator = function() { return 1; };
        model.set('id', 2);
        ok(model.collection.sort.called);
    });

    test('syncArrayField updates an array item field when the array field is set', function() {
        var model = new models.BaseModel();
        model.syncArrayField('field', 'field_x', 0);
        model.set({field: [1, 2]});
        ok(model.get('field_x') === 1);
    });

    test('syncArrayField creates an array item field when the object is initialized', function() {
        var MyModel = models.BaseModel.extend({
            initialize: function() {
                this.syncArrayField('field', 'field_x', 0);
                MyModel.__super__.initialize.apply(this, arguments);
            }
        });
        var model = new MyModel({field: [1, 2]});
        ok(model.get('field_x') === 1);
    });

    test('syncArrayField updates an array field when an array item field is set', function() {
        var model = new models.BaseModel({field: [1, 2]});
        model.syncArrayField('field', 'field_x', 0);
        model.set('field_x', 2);
        ok(model.get('field')[0] === 2);
    });

    test('change marks the model as dirty', function() {
        var model = new models.BaseModel();
        ok(model.dirty === false);

        model.set('thing', 1);
        ok(model.dirty === true);
    });

    test('save sets correct values in options object with no options provided', function() {
        var model = new models.BaseModel();
        var origSave = models.BaseModel.__super__.save;
        models.BaseModel.__super__.save = sinon.spy();
        var opts =  {
            wait: true,
            error: model.error,
            success: model.success
        };

        model.save();

        ok(models.BaseModel.__super__.save.called === true);
        ok(models.BaseModel.__super__.save.calledWith(undefined, opts) === true);

        models.BaseModel.__super__.save = origSave;
    });

    test('save honors options passed into it', function() {
        var model = new models.BaseModel();
        var origSave = models.BaseModel.__super__.save;
        models.BaseModel.__super__.save = sinon.spy();

        function error() { }
        function success() { }

        var opts = {
            wait: false,
            error: error,
            success: success
        };

        model.save(null, opts);

        ok(models.BaseModel.__super__.save.called === true);
        ok(models.BaseModel.__super__.save.calledWith(null, opts) === true);

        models.BaseModel.__super__.save = origSave;
    });

    test('success marks the model as clean and sets errors field to null', function() {
        var model = new models.BaseModel();
        ok(model.dirty === false);

        model.dirty = true;
        model.errors = 'bogus!';

        model.success(model);

        ok(model.dirty === false);
        ok(model.errors === null);
    });

    test('destroy passes default options to superclass method', function() {
        var model = new models.BaseModel();
        var origdestroy = models.BaseModel.__super__.destroy;
        models.BaseModel.__super__.destroy = sinon.spy();
        
        model.destroy();

        var opts = {
            wait: true
        };

        ok(models.BaseModel.__super__.destroy.called === true);
        ok(models.BaseModel.__super__.destroy.calledWith(opts) === true);

        models.BaseModel.__super__.destroy = origdestroy;
    });

    test('destroy honors options passed into it', function() {
        var model = new models.BaseModel();
        var origdestroy = models.BaseModel.__super__.destroy;
        models.BaseModel.__super__.destroy = sinon.spy();

        var opts = {
            wait: false
        };
        
        model.destroy(opts);

        ok(models.BaseModel.__super__.destroy.called === true);
        ok(models.BaseModel.__super__.destroy.calledWith(opts) === true);

        models.BaseModel.__super__.destroy = origdestroy;
    });

    test('error reverts to previous attributes of an object', function() {
        var model = new models.BaseModel();
        var response = {
            responseText: JSON.stringify({
                errors: [{
                    description: "it's broken!"
                }]
            })
        };
        model.set('test', 1);
        model.set('test', 2);
        model.error(model, response);
        ok(model.get('test', 1));
    });

    test('error parses error data from a JSON response', function() {
        var model = new models.BaseModel();
        var response = {
            responseText: JSON.stringify({
                errors: [{
                    description: "it's broken!"
                }]
            })
        };

        model.error(model, response);

        ok(model.errors.length === 1);
        ok(model.errors[0].description === "it's broken!");
    });

    test('error catches a JSON parsing exception and returns a generic error', function() {
        var model = new models.BaseModel();
        var response = {
            responseText: 'oh fudge! {'
        };

        model.error(model, response);

        ok(model.errors.length === 1);
        ok(model.errors[0].description === "A server error prevented saving the model.");
    });

    test('fetch provides a success function by default', function() {
        var model = new models.BaseModel();
        var _fetch = models.BaseModel.__super__.fetch;
        var opts = {
            success: model.success
        };

        models.BaseModel.__super__.fetch = sinon.spy();
        model.fetch();
        ok(models.BaseModel.__super__.fetch.calledWith(opts));
        models.BaseModel.__super__.fetch = _fetch;
    });

    test('parse triggers MESSAGE_RECEIVED if response included a message', function() {
        var model = new models.BaseModel();
        var messageReceived = false;

        model.on(models.BaseModel.MESSAGE_RECEIVED, function() {
            messageReceived = true;
        });

        var response = {
            message: 'oops!'
        };

        model.parse(response);

        ok(messageReceived === true);
    });

    test('parse converts dateFields into moment objects', function() {
        var model = new models.BaseModel();
        model.dateFields = ['date'];
        var date = moment();

        var response = {
            'date': date.toString()
        };

        var data = model.parse(response);

        ok(date.unix() === data.date.unix());
    });

    test('parse ignores dateFields if they are not strings', function() {
        var model = new models.BaseModel();
        model.dateFields = ['date'];
        var date = moment();
        var obj = {one: 1};

        var response = {
            'date': obj
        };

        var data = model.parse(response);

        ok(data.date = obj);
    });

    test('get converts dateFields into moment objects', function() {
        var model = new models.BaseModel();
        model.dateFields = ['date'];
        var date = moment();

        model.set('date', date.toString());

        ok(model.get('date').unix() === date.unix());
    });

    test('get ignores null datefields', function() {
        var model = new models.BaseModel();
        model.dateFields = ['date'];
        var date = moment();

        model.set('date', null);

        ok(model.get('date') === null);
    });

    test('get returns the actual value of a field if it is a dateField and moment.js cannot parse it', function() {
        var model = new models.BaseModel();
        model.dateFields = ['date'];
        var badValue = 'not a date';

        model.set('date', badValue);

        ok(model.get('date') === badValue);
    });

    test('toJSON should convert dateFields into strings', function() {
        var date = moment();
        var model = new models.BaseModel({date: date});
        model.dateFields = ['date'];
        ok(model.toJSON()['date'] === date.format());
    });

    test('url is prepended with GnomeModel URL if gnomeModel is passed into constructor', function() {
        var gnome = new models.GnomeModel({
            id: 'abc'
        });

        var TestModel = models.BaseModel.extend({
            url: '/test_model'
        });

        var model = new TestModel({}, {
            gnomeModel: gnome
        });

        ok(model.url() === '/model/abc/test_model');
    });


    module('BaseCollection');

    test('url is prepended with GnomeModel URL if gnomeModel is passed into constructor', function() {
        var gnome = new models.GnomeModel({
            id: 'abc'
        });

        var TestModel = models.BaseModel.extend({});

        var TestCollection = models.BaseCollection.extend({
            model: TestModel,
            url: '/test_model'
        });

        var collection = new TestCollection([{id: 123}], {
            gnomeModel: gnome
        });

        var model = collection.at(0);

        ok(model.url() === '/model/abc/test_model/123');
    });


    module('Gnome');

    test('url includes ID if present', function() {
        var gnome = new models.GnomeModel();
        ok(gnome.url() == '/model');

        gnome.id = 123;
        ok(gnome.url() === '/model/123');
    });


    module('SurfaceReleaseSpillCollection');

    test('sorting compares by release_time', function() {
        function spill(time) {
            return new models.SurfaceReleaseSpill({release_time: time});
        }

        var now = moment();
        var first = spill(now);
        var second = spill(now.clone().add('hours', 1));
        var third = spill(now.clone().add('hours', 2));
        var coll = new models.SurfaceReleaseSpillCollection([second, third, first]);

        coll.sort();
        ok(coll.models[0] === first);
        ok(coll.models[1] === second);
        ok(coll.models[2] === third);
    });


    module('Wind');

    test('initialize sets timeseries if not provided', function() {
        var wind = new models.Wind();
        var timeseries = wind.get('timeseries');
        var default_timeseries = timeseries[0];
        var now = moment();
        ok(default_timeseries[0].days() === now.days());
        ok(default_timeseries[0].years() === now.years());
        ok(default_timeseries[0].hour() === now.hour());
        ok(default_timeseries[0].minute() === now.minute());
        ok(default_timeseries[1] === 0);
        ok(default_timeseries[2] === 0);
    });

    test('set sorts a timeseries array by datetime (index position 0)', function() {
        var now = moment();
        var first = [now];
        var second = [now.clone().add('hours', 1)];
        var third = [now.clone().add('hours', 2)];
        var wind = new models.Wind({timeseries: [third, first, second]});
        var timeseries = wind.get('timeseries');

        ok(timeseries[0] === first);
        ok(timeseries[1] === second);
        ok(timeseries[2] === third);

        // test the other way to set a field
        wind.set('timeseries', null);
        wind.set('timeseries', [third, first, second]);
        timeseries = wind.get('timeseries');

        ok(timeseries[0] === first);
        ok(timeseries[1] === second);
        ok(timeseries[2] === third);
    });

    test('isNws works', function() {
        var wind = new models.Wind({source_type: 'nws'});
        ok(wind.isNws());
    });

    test('isBuoy works', function() {
        var wind = new models.Wind({source_type: 'buoy'});
        ok(wind.isBuoy());
    });

    test('type reports "constant-wind" if wind has one timeseries value', function() {
        var wind = new models.Wind({timeseries: [1]});
        ok(wind.type() === 'constant-wind');
    });

    test('type reports "variable-wind" if wind has multiple timeseries values', function() {
        var wind = new models.Wind({timeseries: [1, 2]});
        ok(wind.type() === 'variable-wind');
    });

    test('constantSpeed reports the speed of the first wind timeseries', function() {
        var wind = new models.Wind({timeseries: [
            [1, 2, 3]
        ]});
        ok(wind.constantSpeed() === wind.get('timeseries')[0][1]);
    });

    test('constantDirection reports the speed of the first wind timeseries', function() {
        var wind = new models.Wind({timeseries: [
            [1, 2, 3]
        ]});
        ok(wind.constantDirection() === wind.get('timeseries')[0][2]);
    });


    module('RandomMoverCollection', function() {
        var movers = [
            new models.RandomMover({active_start: moment('1/1/2013')}),
            new models.RandomMover({active_start: moment('1/3/2013')}),
            new models.RandomMover({active_start: moment('1/2/2013')}),
        ];
        var collection = new models.WindMoverCollection(movers);

        collection.sort();

        ok(collection.at(0).cid === movers[0].cid);
        ok(collection.at(1).cid === movers[1].cid);
        ok(collection.at(2).cid === movers[2].cid);
    });


    module('Map');

    test('url includes the gnomeModel URL', function() {
        var gnome = new models.GnomeModel({
            id: 123
        });
        var map = new models.Map({}, {
            gnomeModel: gnome
        });
        ok(map.url() === '/model/123/map');
    });


    module('CustomMap');

    test('url includes the gnomeModel URL', function() {
        var gnome = new models.GnomeModel({
            id: 123
        });
        var map = new models.CustomMap({}, {
            gnomeModel: gnome
        });
        ok(map.url() === '/model/123/custom_map');
    });


    module('getNwsWind');

    asyncTest('getNwsWind returns values', function() {
        var coords = {
            longitude: -72.419992,
            latitude: 41.20212
        };
        models.getNwsWind(coords, {
            success: function(data) {
                ok(data.description === "23NM WSW New London CT");
                start();
            }
        })
    });
});


