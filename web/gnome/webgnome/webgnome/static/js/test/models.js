define([
    'models',
    'lib/moment'
], function(models) {

    module('TimeStep');

    test('get should convert timestamp fields to MM/DD/YYY HH:mm format', function() {
        var time = moment().toString();
        var step = new models.TimeStep({timestamp: time});
        ok(step.get('timestamp') == moment(time).format('MM/DD/YYYY HH:mm'));
    });


    module('BaseModel');

    test('get should convert dateFields into moment objects', function() {
        var date = moment();
        var model = new models.BaseModel({date: date.toString()});
        model.dateFields = ['date'];
        ok(date.format() == model.get('date').format());
    });

    test('toJSON should convert dateFields into strings', function() {
        var date = moment();
        var model = new models.BaseModel({date: date});
        model.dateFields = ['date'];
        ok(model.toJSON()['date'] == date.format());
    });
});

