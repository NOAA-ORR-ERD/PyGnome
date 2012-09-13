"""imap.py -- Friendly front-end to imaplib.

IMAP4 RFC is at http://www.ietf.org/rfc/rfc3501.txt

Note: messages that are in Mozilla Firefox's Trash appear in INBOX here.  To
get rid of said messages, empty the trash, then quit Firefox and restart it.

To delete messages, "STORE +Flags (\Deleted)", "EXPUNGE".
"""
import email, imaplib, logging, pprint, re
info = logging.getLogger('imap').info

uid_rx = re.compile( R"UID (\d+)" )

def parse_response(rx, s):
    m = rx.search(s)
    if m:
        return m.group(1)
    else:
        return None

def make_message_set(ids):
    """Convert a list of message IDs to an IMAP-format string.  If 'ids'
       is not a list, return it unchanged.  (Assume it's a single message ID
       or a string of space-separated IDs.)  Any element may be an IMAP-format
       range or a special range like 'ALL'.
    """
    if isinstance(ids, list):
        ids_str = ",".join(ids)
        return "%s" % ids_str
    else:
        return ids

class IMAP(object):
    def __init__(self, host, username, password, ssl=False, port=None, 
        folder='INBOX', readonly=False, uid_mode=False, debug=False):
        """Log in to an IMAP or IMAP_SSL server."""
        if ssl:
            port = port or imaplib.IMAP4_SSL_PORT
            info("opening SSL connection to %s port %s", host, port)
            self.imap = imap = imaplib.IMAP4_SSL(host, port)
        else:
            port = port or imaplib.IMAP4_PORT
            info("opening connection to %s port %s", host, port)
            self.imap = imap = imaplib.IMAP4(host, port)
        self.uid_mode = uid_mode
        if debug is True:
            imap.debug = 4
        elif debug:
            imap.debug = debug
        imap.login(username, password)
        self.select(folder, readonly)

    def select(self, folder, readonly):
        """Select a folder; set instance variables."""
        response = self.imap.select(folder, readonly)
        try:
            self.message_count = int(response[1][0])
        except (TypeError, ValueError):
            self.message_count = None
        self.uidvalidity = self.response('UIDVALIDITY')

    def check_uidvalidity(self, p):
        """Check UIDVALIDITY against a saved value in a file.  
        
           @param p A path object.  (Jason Orendorff's 'path' module.)
           @exc RuntimeError if the values differ.

           If the file does not exist, write the current value to it.  This
           is not an error.
        """
        assert hasattr(self, 'uidvalidity'), ".select() has not been called!"
        if not p.exists():
            info("writing UIDVALIDITY %r to new file %s", self.uidvalidity, p)
            p.write_text(self.uidvalidity)
        elif p.text() != self.uidvalidity:
            m = "UIDVALIDITY has changed! (expected %s, found %s, file %s)"
            m %= (self.uidvalidity, p.text(), p.abspath())
            raise RuntimeError(m)

    def response(self, code):
        """Return a response from a previous command.
        
           @param code str The desired response type.
           @ret str, 2-tuple, or None.
        """
        return self.imap.response(code)[1][0]

    def responses(self, code):
        """Return all 'code' responses from a previous command.

           @param code str The desired response type.
           @ret list of (str or 2-tuple).
        """
        return self.imap.response(code)[1]

    def list_folders(self, directory='""', pattern='*'):
        """Return the names of all available folders.

           I'm not sophistocated enough to parse the folder names,
           so I just return the raw response, which is probably a 
           list of strings.  The folder names are probably the last
           quoted word in each string before the ")", but I don't know
           for sure.
        """
        return self.imap.list(directory, pattern)[1]
        

    def search(self, charset, *criteria):
        """General search.

            @param charset str or None.  (Normally None.)
            @param *criteria str Criteria in IMAP format.
            @ret list of strings Message sequence numbers.
        """
        if self.uid_mode:
            result = self.imap.uid('SEARCH', charset, *criteria)
        else:
            result = self.imap.search(charset, *criteria)
        return result[1][0].split()

    def fetch(self, message_ids, message_parts):
        """General fetch.  Extract message bodies, headers, etc.

           @param message_set str List of message
               sequence numbers.  Each element may be an IMAP-format range.
           @param message_parts str IMAP spec of desired components.
           @ret list of whatever imaplib fetches (strings, tuples,
              lists of either).

          @@MO: Currently broken for multiple message_ids.
        """
        message_set = make_message_set(message_ids)
        #print message_set
        if ' ' in message_set:
            raise NotImplementedError("fetching multiple messages at once")
        if self.uid_mode:
            result = self.imap.uid('FETCH', message_set, message_parts)
        else:
            result = self.imap.fetch(message_set, message_parts)
        return result[1]

    def fetch_headers(self, message_ids, headers):
        """Extract certain RFC822 headers from messages.

           @param message_set See .fetch().
           @param headers List of RFC822 header names.
           @ret list of dicts (header name : value).
           
           If a message contains multiple values for a header,
           only one value will be present in the dict.
        """
        headers_str = ' '.join(headers)
        message_parts = "(BODY[HEADER.FIELDS (%s)])" % headers_str
        data = self.fetch(message_ids, message_parts)[0][1]
        return [email.message_from_string(data)]

    def fetch_raw_message(self, message_id):
        """Return a message in RFC822 format.

           If 'message_id' refers to multiple messages, only the first is
           returned.
        """
        return self.fetch(message_id, "RFC822")[0][1]

    def fetch_uid_and_rfc822(self, message_ids):
        message_parts = "(UID RFC822)"
        data = self.fetch(message_ids, message_parts)
        uid = parse_response(uid_rx, data[0][0])
        assert uid is not None
        rfc822 = data[0][1]
        return [(uid, rfc822)]

#### END IMAP helper class

'''
if not dest_dir.exists():
    dest_dir.mkdir()
existing_messages = set(x for x in os.listdir(dest_dir) if
    x != uidvalidity_file.name)

mail = imap.IMAP(host, username, password, ssl=ssl, port=port, 
    folder='INBOX', readonly=True, debug=debug)
mail.check_uidvalidity(uidvalidity_file)
existing_messages = [x for x in os.listdir(dest_dir) if
    x != uidvalidity_file.name]
# Can't figure out syntax to search for UIDs except those existing,
# so we'll just get all of them and ignore the duplicates.
message_ids = mail.search(None, "ALL")
print message_ids ; sys.exit()
pprint.pprint(mail.fetch(message_ids, "UID")) ;sys.exit()
for message_id in message_ids:
    data = mail.fetch_uid_and_rfc822(message_id)
    uid = data[0][0]
    message = data[0][1]
    p = dest_dir / uid
    if p.exists():
        raise RuntimeError("message collision, UID=%s" % uid)
    else:
        info("writing message to file %s", p)
        p.write_text(message)
'''
