define([
    'models'
], function(models) {

    var origAjax = $.ajax;
    var testUrl = "http://localhost/test";

    module('AjaxForm');

    function makeForm(opts) {
        var options = $.extend({}, opts || {}, {
            url: testUrl
        });
        return new models.AjaxForm(options);
    }

    test('Receiving a message should trigger MESSAGE_RECEIVED', function() {
        var messageReceivedTriggered = false;
        var data = {
            message: {
                text: 'Fake message',
                type: 'test'
            }
        };

        var form = makeForm();

        form.on(models.AjaxForm.MESSAGE_RECEIVED, function() {
            messageReceivedTriggered = true;
        });

        form.parse(data);

        equal(messageReceivedTriggered, true);
    });


    test('Making a request should append the passed-in ID', function() {
        var form = makeForm();
        var calledWith = {};

        $.ajax = function(opts) {
            calledWith = opts;
        };

        form.makeRequest({
            id: 1
        });

        equal(calledWith.url, testUrl + '/1');

        $.ajax = origAjax;
    });

    test('Get should use GET request type', function() {
        var form = makeForm(),
            calledWith = {};

        $.ajax = function(opts) {
            calledWith = opts;
        };

        form.get({
            id: 1
        });

        equal(calledWith.url, testUrl + '/1');
        equal(calledWith.type, 'GET');

        $.ajax = origAjax;
    });

    test('Submit should use POST request type', function() {
        var form = makeForm(),
            calledWith = {};

        $.ajax = function(opts) {
            calledWith = opts;
        };

        form.submit({
            id: 1
        });

        equal(calledWith.url, testUrl + '/1');
        equal(calledWith.type, 'POST');

        $.ajax = origAjax;
    });
});
