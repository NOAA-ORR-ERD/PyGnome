/*	Add scroll support to divs
	Based off of Chris Barr's solution
	http://chris-barr.com/files/touchScroll.htm	*/
(function($){
	$.fn.extend({ 
	touchScroll: function() {
			// Detect if is touchscreen device, if it is not simply return
			if ((navigator.userAgent.match(/android 3/i)) || (navigator.userAgent.match(/honeycomb/i))) {
				return this;
			}
			try {
				document.createEvent("TouchEvent");
			} catch (e) {
				return this;
			}
			return this.each(function() {
				var el = this, scrollStartPosY = 0, scrollStartPosX = 0;
				el.addEventListener("touchstart", function (event) {
					scrollStartPosY = this.scrollTop + event.touches[0].pageY;
					scrollStartPosX = this.scrollLeft + event.touches[0].pageX;
					
				}, false);
				el.addEventListener("touchmove", function (event) {
					
					if ((this.scrollTop < this.scrollHeight - this.offsetHeight && this.scrollTop + event.touches[0].pageY < scrollStartPosY - 5) || (this.scrollTop !== 0 && this.scrollTop + event.touches[0].pageY > scrollStartPosY + 5)) {
						event.preventDefault();
					}
					if ((this.scrollLeft < this.scrollWidth - this.offsetWidth && this.scrollLeft + event.touches[0].pageX < scrollStartPosX - 5) || (this.scrollLeft !== 0 && this.scrollLeft + event.touches[0].pageX > scrollStartPosX + 5)) {
						event.preventDefault();
					}
					this.scrollTop = scrollStartPosY - event.touches[0].pageY;
					this.scrollLeft = scrollStartPosX - event.touches[0].pageX;
				}, false);
			});
		}
	});
})(jQuery);