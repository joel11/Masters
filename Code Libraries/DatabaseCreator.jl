#Pkg.add("SQLite")
using SQLite

function CreateDatabase(database_name)

    db = SQLite.DB(database_name)

    configuration_run_table =
    " CREATE TABLE IF NOT EXISTS configuration_run(
        configuration_id INTEGER PRIMARY KEY AUTOINCREMENT,
        seed_used INTEGER NOT NULL,
        start_time DATETIME NOT NULL)
        "

    network_parameters_table = "
        CREATE TABLE IF NOT EXISTS network_parameters(
            configuration_id INTEGER PRIMARY KEY,
            category VARCHAR NOT NULL,
            layer_sizes VARCHAR NOT NULL,
            layer_activations VARCHAR NOT NULL,
            initialization VARCHAR NOT NULL)
            "
    training_parameters_table = "
        CREATE TABLE IF NOT EXISTS training_parameters(
            configuration_id INTEGER PRIMARY KEY,
            category VARCHAR NOT NULL,
            learning_rate FLOAT NOT NULL,
            minibatch_size INTEGER NOT NULL,
            max_epochs INTEGER NOT NULL,
            l1_lambda FLOAT,
            l2_lambda FLOAT,
            cost_function VARCHAR,
            stopping_function VARCHAR)"



    epoch_records_table = "
        CREATE TABLE IF NOT EXISTS epoch_records(
            configuration_id INTEGER PRIMARY KEY,
            category VARCHAR NOT NULL,
            record_time DATETIME NOT NULL,
            epoch_number INTEGER NOT NULL,
            mean_minibatch_cost FLOAT NOT NULL,
            training_cost FLOAT NOT NULL,
            testing_cost FLOAT NOT NULL,
            run_time FLOAT NOT NULL)"


    epoch_dataconfig_table = "
        CREATE TABLE IF NOT EXISTS dataset_config(
            configuration_id INTEGER PRIMARY KEY,
            category VARCHAR NOT NULL,
            steps INTEGER NOT NULL,
            deltas VARCHAR NOT NULL,
            process_splits VARCHAR,
            training_splits VARCHAR,
            prediction_steps VARCHAR NOT NULL,
            variation_values VARCHAR)
    "

    SQLite.execute!(db, configuration_run_table)
    SQLite.execute!(db, training_parameters_table)
    SQLite.execute!(db, network_parameters_table)
    SQLite.execute!(db, epoch_records_table)
    SQLite.execute!(db, epoch_dataconfig_table)
end

#CreateDatabase("database_test")
