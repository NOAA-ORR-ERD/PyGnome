import sys, os
import transaction
from .oil_library_parse import LibFile

from sqlalchemy import engine_from_config

from pyramid.paster import (
    get_appsettings,
    setup_logging,
    )

from .models import (
    DBSession,
    Base,
    Oil,
    Synonym,
    Density,
    KVis,
    DVis,
    Cut,
    Toxicity,
    )

def usage(argv):
    cmd = os.path.basename(argv[0])
    print('usage: %s <config_uri>\n'
          '(example: "%s development.ini")' % (cmd, cmd)) 
    sys.exit(1)

def main():
    if len(sys.argv) != 2:
        usage(sys.argv)
    config_uri = sys.argv[1]
    setup_logging(config_uri)
    settings = get_appsettings(config_uri)
    engine = engine_from_config(settings, 'sqlalchemy.')
    DBSession.configure(bind=engine)
    Base.metadata.create_all(engine) #@UndefinedVariableFromImport
    with transaction.manager:
        # -- Our loading routine --
        session = DBSession()

        # 1. purge our builtin rows if any exist
        sys.stderr.write('Purging old records in database')
        oilobjs = session.query(Oil).filter(Oil.custom==False)
        rowcount = 0
        for o in oilobjs:
            session.delete(o)
            if rowcount % 100 == 0:
                sys.stderr.write('.')
            rowcount += 1
        transaction.commit()
        print 'finished!!!  %d rows processed.' % (rowcount)

        # 2. we need to open our OilLib file
        print 'opening file: %s ...' % (settings['oillib.file'])
        fd = LibFile(settings['oillib.file'])
        print 'file version:', fd.__version__

        # 3. iterate over our rows
        sys.stderr.write('Adding new records to database')
        rowcount = 0
        for r in fd.readlines():
            # 3a. for each row, we populate the Oil object
            initargs = dict(zip(fd.file_columns, r))
            transaction.begin()
            oil = Oil(**initargs) #IGNORE:W0142

            if initargs.get('Synonyms'):
                for s in initargs.get('Synonyms').split(','):
                    s = s.strip()
                    if len(s) > 0:
                        synonyms = session.query(Synonym).filter(Synonym.name==s).all()
                        if len(synonyms) > 0:
                            # we link the existing synonym object
                            oil.synonyms.append(synonyms[0]) #IGNORE:E1101
                        else:
                            # we add a new synonym object
                            oil.synonyms.append(Synonym(s)) #IGNORE:E1101

            for i in range(1,5):
                kg_m3 = 'Density#%d (kg/m^3)' % (i)
                ref_temp = 'Density#%d Ref Temp (K)' % (i)
                w = 'Density#%d Weathering' % (i)
                if initargs.get(kg_m3) or initargs.get(ref_temp) or initargs.get(w):
                    densityargs = {}
                    densityargs[kg_m3[10:]] = initargs.get(kg_m3)
                    densityargs[ref_temp[10:]] = initargs.get(ref_temp)
                    densityargs[w[10:]] = initargs.get(w)
                    #print densityargs
                    oil.densities.append(Density(**densityargs)) #IGNORE:E1101

            for i in range(1,7):
                m2_s = 'KVis#%d (m^2/s)' % (i)
                ref_temp = 'KVis#%d Ref Temp (K)' % (i)
                w = 'KVis#%d Weathering' % (i)
                if initargs.get(m2_s) or initargs.get(ref_temp) or initargs.get(w):
                    kvisargs = {}
                    kvisargs[m2_s[7:]] = initargs.get(m2_s)
                    kvisargs[ref_temp[7:]] = initargs.get(ref_temp)
                    kvisargs[w[7:]] = initargs.get(w)
                    #print kvisargs
                    oil.kvis.append(KVis(**kvisargs)) #IGNORE:E1101

            for i in range(1,7):
                kg_ms = 'DVis#%d (kg/ms)' % (i)
                ref_temp = 'DVis#%d Ref Temp (K)' % (i)
                w = 'DVis#%d Weathering' % (i)
                if initargs.get(kg_ms) or initargs.get(ref_temp) or initargs.get(w):
                    dvisargs = {}
                    dvisargs[kg_ms[7:]] = initargs.get(kg_ms)
                    dvisargs[ref_temp[7:]] = initargs.get(ref_temp)
                    dvisargs[w[7:]] = initargs.get(w)
                    #print dvisargs
                    oil.dvis.append(DVis(**dvisargs)) #IGNORE:E1101

            for i in range(1,16):
                vapor_temp = 'Cut#%d Vapor Temp (K)' % (i)
                liquid_temp = 'Cut#%d Liquid Temp (K)' % (i)
                fraction = 'Cut#%d Fraction' % (i)
                if initargs.get(vapor_temp) or initargs.get(liquid_temp) or initargs.get(fraction):
                    cutargs = {}
                    lbl_offset = len(str(i)) + 5
                    cutargs[vapor_temp[lbl_offset:]] = initargs.get(vapor_temp)
                    cutargs[liquid_temp[lbl_offset:]] = initargs.get(liquid_temp)
                    cutargs[fraction[lbl_offset:]] = initargs.get(fraction)
                    #print cutargs
                    oil.cuts.append(Cut(**cutargs)) #IGNORE:E1101

            for i in range(1,4):
                species = 'Tox_EC(%d)Species' % (i)
                hour24 = 'Tox_EC(%d)24h' % (i)
                hour48 = 'Tox_EC(%d)48h' % (i)
                hour96 = 'Tox_EC(%d)96h' % (i)
                if initargs.get(species) or initargs.get(hour24) or initargs.get(hour48) or initargs.get(hour96):
                    toxargs = {}
                    lbl_offset = len(str(i)) + 8
                    toxargs['Toxicity Type'] = 'EC'
                    toxargs[species[lbl_offset:]] = initargs.get(species)
                    toxargs[hour24[lbl_offset:]] = initargs.get(hour24)
                    toxargs[hour48[lbl_offset:]] = initargs.get(hour48)
                    toxargs[hour96[lbl_offset:]] = initargs.get(hour96)
                    #print toxargs
                    oil.toxicities.append(Toxicity(**toxargs)) #IGNORE:E1101

            for i in range(1,4):
                species = 'Tox_LC(%d)Species' % (i)
                hour24 = 'Tox_LC(%d)24h' % (i)
                hour48 = 'Tox_LC(%d)48h' % (i)
                hour96 = 'Tox_LC(%d)96h' % (i)
                if initargs.get(species) or initargs.get(hour24) or initargs.get(hour48) or initargs.get(hour96):
                    toxargs = {}
                    lbl_offset = len(str(i)) + 8
                    toxargs['Toxicity Type'] = 'LC'
                    toxargs[species[lbl_offset:]] = initargs.get(species)
                    toxargs[hour24[lbl_offset:]] = initargs.get(hour24)
                    toxargs[hour48[lbl_offset:]] = initargs.get(hour48)
                    toxargs[hour96[lbl_offset:]] = initargs.get(hour96)
                    #print toxargs
                    oil.toxicities.append(Toxicity(**toxargs)) #IGNORE:E1101

            session.add(oil)
            transaction.commit()
            
            if rowcount % 100 == 0:
                sys.stderr.write('.')
            rowcount += 1
        print 'finished!!!  %d rows processed.' % (rowcount)
        # end 'with transaction.manager:'

