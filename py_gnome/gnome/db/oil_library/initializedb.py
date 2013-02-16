import sys, os
import transaction
from .oil_library_parse import OilLibraryFile

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

def initialize_sql(settings):
    engine = engine_from_config(settings, 'sqlalchemy.')
    DBSession.configure(bind=engine)
    Base.metadata.create_all(engine) #@UndefinedVariableFromImport

def load_database(settings):
    with transaction.manager:
        # -- Our loading routine --
        session = DBSession()

        # 1. purge our builtin rows if any exist
        sys.stderr.write('Purging old records in database')
        num_purged = purge_old_records(session)
        print 'finished!!!  %d rows processed.' % (num_purged)

        # 2. we need to open our OilLib file
        print 'opening file: %s ...' % (settings['oillib.file'])
        fd = OilLibraryFile(settings['oillib.file'])
        print 'file version:', fd.__version__

        # 3. iterate over our rows
        sys.stderr.write('Adding new records to database')
        rowcount = 0
        for r in fd.readlines():
            # 3a. for each row, we populate the Oil object
            add_oil_object(session, fd.file_columns, r)    
            if rowcount % 100 == 0:
                sys.stderr.write('.')
            rowcount += 1
        print 'finished!!!  %d rows processed.' % (rowcount)

def purge_old_records(session):
    oilobjs = session.query(Oil).filter(Oil.custom==False)
    rowcount = 0
    for o in oilobjs:
        session.delete(o)
        if rowcount % 100 == 0:
            sys.stderr.write('.')
        rowcount += 1
    transaction.commit()
    return rowcount

def add_oil_object(session, file_columns, row_data):
    row_dict = dict(zip(file_columns, row_data))
    transaction.begin()
    oil = Oil(**row_dict) #IGNORE:W0142

    add_synonyms(session, oil, row_dict)
    add_densities(oil, row_dict)
    add_kinematic_viscosities(oil, row_dict)
    add_dynamic_viscosities(oil, row_dict)
    add_distillation_cuts(oil, row_dict)
    add_toxicity_effective_concentrations(oil, row_dict)
    add_toxicity_lethal_concentrations(oil, row_dict)

    session.add(oil)
    transaction.commit()

def add_synonyms(session, oil, row_dict):
    if row_dict.get('Synonyms'):
        for s in row_dict.get('Synonyms').split(','):
            s = s.strip()
            if len(s) > 0:
                synonyms = session.query(Synonym).filter(Synonym.name==s).all()
                if len(synonyms) > 0:
                    # we link the existing synonym object
                    oil.synonyms.append(synonyms[0]) #IGNORE:E1101
                else:
                    # we add a new synonym object
                    oil.synonyms.append(Synonym(s)) #IGNORE:E1101

def add_densities(oil, row_dict):
    for i in range(1,5):
        kg_m3 = 'Density#%d (kg/m^3)' % (i)
        ref_temp = 'Density#%d Ref Temp (K)' % (i)
        w = 'Density#%d Weathering' % (i)
        if row_dict.get(kg_m3) or row_dict.get(ref_temp) or row_dict.get(w):
            densityargs = {}
            densityargs[kg_m3[10:]] = row_dict.get(kg_m3)
            densityargs[ref_temp[10:]] = row_dict.get(ref_temp)
            densityargs[w[10:]] = row_dict.get(w)
            #print densityargs
            oil.densities.append(Density(**densityargs)) #IGNORE:E1101

def add_kinematic_viscosities(oil, row_dict):
    for i in range(1,7):
        m2_s = 'KVis#%d (m^2/s)' % (i)
        ref_temp = 'KVis#%d Ref Temp (K)' % (i)
        w = 'KVis#%d Weathering' % (i)
        if row_dict.get(m2_s) or row_dict.get(ref_temp) or row_dict.get(w):
            kvisargs = {}
            kvisargs[m2_s[7:]] = row_dict.get(m2_s)
            kvisargs[ref_temp[7:]] = row_dict.get(ref_temp)
            kvisargs[w[7:]] = row_dict.get(w)
            #print kvisargs
            oil.kvis.append(KVis(**kvisargs)) #IGNORE:E1101

def add_dynamic_viscosities(oil, row_dict):
    for i in range(1,7):
        kg_ms = 'DVis#%d (kg/ms)' % (i)
        ref_temp = 'DVis#%d Ref Temp (K)' % (i)
        w = 'DVis#%d Weathering' % (i)
        if row_dict.get(kg_ms) or row_dict.get(ref_temp) or row_dict.get(w):
            dvisargs = {}
            dvisargs[kg_ms[7:]] = row_dict.get(kg_ms)
            dvisargs[ref_temp[7:]] = row_dict.get(ref_temp)
            dvisargs[w[7:]] = row_dict.get(w)
            #print dvisargs
            oil.dvis.append(DVis(**dvisargs)) #IGNORE:E1101

def add_distillation_cuts(oil, row_dict):
    for i in range(1,16):
        vapor_temp = 'Cut#%d Vapor Temp (K)' % (i)
        liquid_temp = 'Cut#%d Liquid Temp (K)' % (i)
        fraction = 'Cut#%d Fraction' % (i)
        if row_dict.get(vapor_temp) or row_dict.get(liquid_temp) or row_dict.get(fraction):
            cutargs = {}
            lbl_offset = len(str(i)) + 5
            cutargs[vapor_temp[lbl_offset:]] = row_dict.get(vapor_temp)
            cutargs[liquid_temp[lbl_offset:]] = row_dict.get(liquid_temp)
            cutargs[fraction[lbl_offset:]] = row_dict.get(fraction)
            #print cutargs
            oil.cuts.append(Cut(**cutargs)) #IGNORE:E1101

def add_toxicity_effective_concentrations(oil, row_dict):
    for i in range(1,4):
        species = 'Tox_EC(%d)Species' % (i)
        hour24 = 'Tox_EC(%d)24h' % (i)
        hour48 = 'Tox_EC(%d)48h' % (i)
        hour96 = 'Tox_EC(%d)96h' % (i)
        if row_dict.get(species) or row_dict.get(hour24) or row_dict.get(hour48) or row_dict.get(hour96):
            toxargs = {}
            lbl_offset = len(str(i)) + 8
            toxargs['Toxicity Type'] = 'EC'
            toxargs[species[lbl_offset:]] = row_dict.get(species)
            toxargs[hour24[lbl_offset:]] = row_dict.get(hour24)
            toxargs[hour48[lbl_offset:]] = row_dict.get(hour48)
            toxargs[hour96[lbl_offset:]] = row_dict.get(hour96)
            #print toxargs
            oil.toxicities.append(Toxicity(**toxargs)) #IGNORE:E1101

def add_toxicity_lethal_concentrations(oil, row_dict):
    for i in range(1,4):
        species = 'Tox_LC(%d)Species' % (i)
        hour24 = 'Tox_LC(%d)24h' % (i)
        hour48 = 'Tox_LC(%d)48h' % (i)
        hour96 = 'Tox_LC(%d)96h' % (i)
        if row_dict.get(species) or row_dict.get(hour24) or row_dict.get(hour48) or row_dict.get(hour96):
            toxargs = {}
            lbl_offset = len(str(i)) + 8
            toxargs['Toxicity Type'] = 'LC'
            toxargs[species[lbl_offset:]] = row_dict.get(species)
            toxargs[hour24[lbl_offset:]] = row_dict.get(hour24)
            toxargs[hour48[lbl_offset:]] = row_dict.get(hour48)
            toxargs[hour96[lbl_offset:]] = row_dict.get(hour96)
            #print toxargs
            oil.toxicities.append(Toxicity(**toxargs)) #IGNORE:E1101



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
    initialize_sql(settings)    
    load_database(settings)
