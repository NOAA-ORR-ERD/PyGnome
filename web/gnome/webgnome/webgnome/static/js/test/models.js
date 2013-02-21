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

    test('get should convert dateFields into moment objects', function() {
        var date = moment();
        var model = new models.BaseModel({date: date.toString()});
        model.dateFields = ['date'];
        ok(date.format() === model.get('date').format());
    });

    test('toJSON should convert dateFields into strings', function() {
        var date = moment();
        var model = new models.BaseModel({date: date});
        model.dateFields = ['date'];
        ok(model.toJSON()['date'] === date.format());
    });

    test('collection is resorted if ID field changes', function() {
        var model = new models.BaseModel({id: 1});
        model.collection = {};
        model.collection.sort = sinon.spy();
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

    test('error parses error data from a JSON response', function() {
        var model = new models.BaseModel();
        model.set('test', 1);
        model.set('test', 2);
        var response = {
            responseText: JSON.stringify({
                errors: [{
                    text: "it's broken!"
                }]
            })
        };

        model.error(model, response);

        ok(model.errors.length === 1);
        ok(model.errors[0].text === "it's broken!");
    });
});

