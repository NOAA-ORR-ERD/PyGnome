/*
 *  Compass UI - a javascript control for selecting velocity and direction
 *               for wind and water.
 */
(function( $ ) {
  $.fn.compassUI = function(options) {

    return this.each(function() {

      // Public API method: clear the front canvas (arrow and line shapes)
      if (options === 'reset') {
          this.frontcanv.getContext('2d').clearRect(
              0, 0, this.frontcanv.width, this.frontcanv.height);
          return;
      }

      this.settings = $.extend( {
        'arrow-direction' : 'out',
        // we can optionally set a function which translates the angle to its nearest cardinal value
        'cardinal-name' : function(angle) {
          var dirNames = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];
          return dirNames[Math.floor((+(angle)+360/32)/(360/16.0)%16)];
        },
        // we can optionally set a function which translates the cardinal direction to its angle
        'cardinal-angle' : function(name) {
          var dirNames = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];
          var idx = dirNames.indexOf(name.toUpperCase())
          if (idx === -1) {
            return null;
          }
          else {
            return (360.0/16)*idx
          }
        },
        // we can optionally set a function which happens on a move
        'move' : null,
        // we can optionally set a function which happens after a mouse drag
        'change' : null,
      }, options);

      $(this).load(function(ev) {
        var event = ev;
      });

      // check if our object is hidden
      $(this).css({'visibility':'hidden', 'display':'block'});




      // here we create our background canvas
      if (this['backcanv'] === undefined) {
        $(this).append('<canvas id="' + this.id + '-back"></canvas>');
        this.backcanv = $('canvas#' + this.id + '-back')[0];
        if (window['G_vmlCanvasManager'] != undefined) {
          // older versions of IE don't natively have canvas functionality
          G_vmlCanvasManager.initElement(backcanv);
        }
      }
      var backcanv = this.backcanv;
      backcanv.width = $(this).width();
      backcanv.height = $(this).height();
      backcanv.style.zIndex = "0";
      backcanv.style.position = 'absolute';

      // here we create our foreground canvas
      if (this['frontcanv'] === undefined) {
        $(this).append('<canvas id="' + this.id + '-front"></canvas>');
        this.frontcanv = $('canvas#' + this.id + '-front')[0];
        if (window['G_vmlCanvasManager'] != undefined) {
          G_vmlCanvasManager.initElement(frontcanv);
        }
      }
      var frontcanv = this.frontcanv;
      frontcanv.width = $(this).width();
      frontcanv.height = $(this).height();
      frontcanv.style.zIndex = "1";
      frontcanv.style.position = 'absolute';

      // if our div was hidden, return it to its original state
      $(this).removeAttr('style');

      //
      // here we draw the background canvas
      //
      var ctx = backcanv.getContext('2d');
      var maxradius = ((backcanv.width > backcanv.height) ? backcanv.height/2-1: backcanv.width/2-1)* 0.75;

      // here is the backround white target
      ctx.beginPath();
      ctx.arc(backcanv.width/2, backcanv.height/2,
                   maxradius,
                   0, Math.PI*2, true);
      ctx.closePath();
      ctx.fillStyle = 'rgba(255, 255, 255, .8)'
      ctx.fill();

      // here are the concentric circles in the target
      frontcanv.px_per_unit = maxradius/50;
      ctx.beginPath();
      for (var i = maxradius/5; i <= maxradius; i += maxradius/5) {
        ctx.moveTo(backcanv.width/2+i, backcanv.height/2);
        ctx.arc(backcanv.width/2, backcanv.height/2,
                i,
                0, 2*Math.PI);
      }
      ctx.closePath();
      ctx.stroke();

      // here are the direction indicator letters
      ctx.fillStyle = 'rgba(0, 0, 0, .8)'
      ctx.translate(backcanv.width/2, backcanv.height/2);
      var fontsize = backcanv.height/20+1;
      var fontpad = backcanv.height/50;
      if (window['G_vmlCanvasManager'] != undefined) {
        ctx.font= 'bold ' + fontsize + 'px Optimer';
      }
      else {
        ctx.font= 'bold ' + fontsize + 'px Times New Roman';
      }
      ctx.fillText('N', -5, -((backcanv.height/2)-fontsize));
      ctx.fillText('S', -5, (backcanv.height/2)-fontpad);
      ctx.fillText('W', -(backcanv.width/2-fontpad), 5);
      ctx.fillText('E', (backcanv.width/2-fontsize), 5);

      var numAngles = 8;
      for (var i = 0; i < numAngles; i++) {
        var angleSize = 360/numAngles;
        var txt = (angleSize*i).toString();
        var txtWidth = ctx.measureText(txt).width;
        ctx.fillText(txt, -txtWidth/2, -(backcanv.height/2-(fontsize*2)-fontpad+1));
        ctx.rotate(angleSize*Math.PI/180);
      }


      //
      // here we draw the front canvas
      // which we will be drawing on
      //
      ctx = frontcanv.getContext('2d');
      ctx.fillRect(20,20,150,100);
      ctx.clearRect(20,20,150,100);

      frontcanv.pressed = false;
      frontcanv.moved = false;

      $(frontcanv).mousedown(function (ev) {
        this.pressed = true;
        if (ev.originalEvent['layerX'] != undefined) {
          this.x0 = ev.originalEvent.layerX;
          this.y0 = ev.originalEvent.layerY;
        }
        else {
          // in IE, we use this property
          this.x0 = ev.originalEvent.x;
          this.y0 = ev.originalEvent.y;
        }
      });

      $(frontcanv).mousemove(function (ev) {
        if (!this.pressed) {
          return;
        }
        this.moved = true;
        var ctx = this.getContext('2d');
        var xcurr, ycurr;
        if (ev.originalEvent['layerX'] != undefined) {
          xcurr = ev.originalEvent.layerX;
          ycurr = ev.originalEvent.layerY;
        }
        else {
          // in IE, we use this property
          xcurr = ev.originalEvent.x;
          ycurr = ev.originalEvent.y;
        }

        var xmag = (xcurr - this.width/2);
        var ymag = -(ycurr - this.height/2);
        this.pmag = Math.sqrt(Math.pow(xmag, 2) + Math.pow(ymag, 2));
        this.pmag /= this.px_per_unit;
        this.pangle = Math.atan2(xmag, ymag)*180/Math.PI;
        if (this.pangle < 0) {
          this.pangle += 360;
        }

        // draw a line from the center
        ctx.lineWidth=2;
        ctx.clearRect(0, 0, this.width, this.height);
        ctx.beginPath();
        ctx.moveTo(this.width/2, this.height/2);
        ctx.lineTo(xcurr, ycurr);
        ctx.stroke();
        ctx.closePath();

        // draw our arrow point
        ctx.beginPath();
        if (this.parentElement.settings && this.parentElement.settings['arrow-direction'] === 'in') {
          ctx.translate(this.width/2, this.height/2);
          ctx.rotate( (this.pangle + 180) *Math.PI/180);
        }
        else {
          ctx.translate(xcurr, ycurr);
          ctx.rotate( this.pangle *Math.PI/180);
        }
        ctx.moveTo(0, 0);
        ctx.quadraticCurveTo(0, 8, 8, 15);
        ctx.lineTo(0, 8);
        ctx.moveTo(0, 0);
        ctx.quadraticCurveTo(0, 8, -8, 15);
        ctx.lineTo(0, 8);
        ctx.stroke();
        ctx.closePath();
        if (this.parentElement.settings && this.parentElement.settings['arrow-direction'] === 'in') {
          ctx.rotate( -(this.pangle + 180) *Math.PI/180);
          ctx.translate(-this.width/2, -this.height/2);
        }
        else {
          ctx.rotate( -this.pangle *Math.PI/180);
          ctx.translate(-xcurr, -ycurr);
        }


        // pass our values to the configured move function
        if (this.parentElement.settings && this.parentElement.settings['move'] != null) {
          this.parentElement.settings['move'](this.pmag, this.pangle);
        }
        ctx.beginPath();
        ctx.closePath();
      });

      $(frontcanv).mouseup(function (ev) {
        if (this.pressed && this.moved) {

          // pass our values to the configured change function
          if (this.parentElement.settings && this.parentElement.settings['change'] != null) {
            this.parentElement.settings['change'](this.pmag, this.pangle);
          }
        }
        this.pressed = this.moved = false;
      });
    });

  };
})( jQuery );

